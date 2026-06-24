"""Anthropic provider scraper.

Two live sources, joined on the model's display name:

* ``GET /v1/models`` (JSON API, requires ``ANTHROPIC_API_KEY``) — model ids,
  display names, token limits, and capabilities.
* The public pricing page (HTML) — per-MTok input/output and cache pricing,
  which is not exposed by the API.

Pricing is best-effort: if the page layout changes or a model isn't listed, the
model is still emitted without a ``pricing`` block.
"""

from __future__ import annotations

import os
import re
import time

import httpx
from selectolax.parser import HTMLParser

from ..base import Scraper
from ..models import Model, Pricing, Provider
from ..tools import endpoint_for

API_URL = "https://api.anthropic.com/v1/models"
API_VERSION = "2023-06-01"
PRICING_URL = "https://docs.claude.com/en/docs/about-claude/pricing"

# The pricing page can return truncated HTML; retry until the parse yields a
# plausible number of models.
PRICING_RETRIES = 4
PRICING_RETRY_DELAY = 0.5  # seconds, multiplied by attempt number (linear backoff)
MIN_PRICED_MODELS = 5

# Pricing table columns we care about, mapped to where they land in our model.
# Everything except input/output goes into Pricing.extra.
_PRICE_COLUMNS = {
    "base input tokens": "input",
    "output tokens": "output",
    "5m cache writes": "cache_write_5m",
    "1h cache writes": "cache_write_1h",
    "cache hits & refreshes": "cache_read",
}

_MONEY = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]+)?)")


class AnthropicScraper(Scraper):
    name = "anthropic"

    def scrape(self) -> Provider:
        pricing_by_name = self._fetch_pricing()
        models: list[Model] = []
        for raw in self._fetch_models():
            display_name = raw.get("display_name")
            models.append(
                Model(
                    id=raw["id"],
                    display_name=display_name,
                    context_window=raw.get("max_input_tokens"),
                    max_output_tokens=raw.get("max_tokens"),
                    modalities=_modalities(raw.get("capabilities", {})),
                    capabilities=_capabilities(raw.get("capabilities", {})),
                    pricing=pricing_by_name.get(_norm(display_name)) if display_name else None,
                )
            )
        return Provider(
            name=self.name,
            root_url="https://api.anthropic.com",
            endpoints=[endpoint_for(self.name, "messages", "/v1/messages")],
            models=models,
        )

    def _fetch_models(self) -> list[dict]:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required to scrape the Anthropic models API"
            )
        headers = {"x-api-key": key, "anthropic-version": API_VERSION}
        out: list[dict] = []
        after_id: str | None = None
        while True:
            params = {"limit": 100}
            if after_id:
                params["after_id"] = after_id
            resp = self._client.get(API_URL, headers=headers, params=params)
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data", [])
            out.extend(data)
            if not body.get("has_more") or not data:
                break
            after_id = data[-1]["id"]
        return out

    def _fetch_pricing(self) -> dict[str, Pricing]:
        """Fetch and parse the pricing page, retrying on failure or a short parse.

        The page occasionally returns truncated HTML, yielding only a handful of
        rows; rather than silently blank most prices, retry until at least
        ``MIN_PRICED_MODELS`` are parsed, then keep the best result seen.
        """
        best: dict[str, Pricing] = {}
        for attempt in range(PRICING_RETRIES):
            try:
                resp = self._client.get(PRICING_URL)
                resp.raise_for_status()
                parsed = _parse_pricing_table(resp.text)
            except httpx.HTTPError:
                parsed = {}
            if len(parsed) > len(best):
                best = parsed
            if len(best) >= MIN_PRICED_MODELS:
                break
            if attempt < PRICING_RETRIES - 1:
                time.sleep(PRICING_RETRY_DELAY * (attempt + 1))
        return best


def _parse_pricing_table(html: str) -> dict[str, Pricing]:
    """Find the model pricing table and map normalized display name -> Pricing."""
    tree = HTMLParser(html)
    for table in tree.css("table"):
        rows = table.css("tr")
        if not rows:
            continue
        header = [c.text(strip=True).lower() for c in rows[0].css("th,td")]
        if "model" not in header:
            continue
        col_index = {h: i for i, h in enumerate(header)}
        if not any(c in col_index for c in _PRICE_COLUMNS):
            continue

        result: dict[str, Pricing] = {}
        for row in rows[1:]:
            cells = [c.text(strip=True) for c in row.css("th,td")]
            if len(cells) != len(header):
                continue
            name = cells[col_index["model"]]
            fields: dict[str, float] = {}
            for column, field in _PRICE_COLUMNS.items():
                idx = col_index.get(column)
                if idx is None:
                    continue
                value = _money(cells[idx])
                if value is not None:
                    fields[field] = value
            if not fields:
                continue
            extra = {k: v for k, v in fields.items() if k not in ("input", "output")}
            result[_norm(name)] = Pricing(
                input=fields.get("input"),
                output=fields.get("output"),
                extra=extra,
            )
        if result:
            return result
    return {}


def _money(text: str) -> float | None:
    m = _MONEY.search(text)
    return float(m.group(1)) if m else None


def _modalities(caps: dict) -> list[str]:
    modalities = ["text"]
    if _supported(caps, "image_input"):
        modalities.append("image")
    if _supported(caps, "pdf_input"):
        modalities.append("pdf")
    return modalities


def _capabilities(caps: dict) -> list[str]:
    return sorted(name for name in caps if _supported(caps, name))


def _supported(caps: dict, name: str) -> bool:
    entry = caps.get(name)
    if isinstance(entry, dict):
        return bool(entry.get("supported"))
    return bool(entry)


def _norm(name: str | None) -> str:
    """Normalize a model display name for joining API and pricing data.

    Drops parenthetical annotations like "(limited availability)" and collapses
    whitespace/case so "Claude Opus 4.8" matches across both sources.
    """
    if not name:
        return ""
    name = re.sub(r"\(.*?\)", "", name)
    return " ".join(name.split()).lower()

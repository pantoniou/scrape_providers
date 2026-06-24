"""DeepSeek provider scraper — fully native (no OpenRouter).

Two native sources:

* ``GET https://api.deepseek.com/models`` (requires ``DEEPSEEK_API_KEY``) — the
  authoritative list of model ids DeepSeek actually serves.
* The docs pricing page (``api-docs.deepseek.com``) — a transposed table (models
  as columns) giving cache-hit / cache-miss input pricing, output pricing,
  context length, and max output for every served model.

DeepSeek caching is automatic, so cache-hit (``cache_read``) pricing is published
for all their models; this is why we scrape DeepSeek directly rather than relying
on OpenRouter's spotty per-host cache figures.
"""

from __future__ import annotations

import os
import re

import httpx
from selectolax.parser import HTMLParser

from ..base import Scraper
from ..models import Model, Pricing, Provider
from ..tools import endpoint_for

MODELS_URL = "https://api.deepseek.com/models"
PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing"

_MONEY = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")
_FOOTNOTE = re.compile(r"\(.*?\)|\s+")  # strip "(1)" footnotes / whitespace from ids


class DeepSeekScraper(Scraper):
    name = "deepseek"

    def scrape(self) -> Provider:
        details = self._fetch_pricing()  # id -> parsed characteristics + pricing
        models: list[Model] = []
        for model_id in self._fetch_model_ids():
            d = details.get(model_id, {})
            models.append(
                Model(
                    id=model_id,
                    display_name=d.get("display_name"),
                    context_window=d.get("context_window"),
                    max_output_tokens=d.get("max_output_tokens"),
                    modalities=["text"],
                    capabilities=d.get("capabilities", []),
                    open_source=True,
                    pricing=d.get("pricing"),
                )
            )
        return Provider(
            name=self.name,
            root_url="https://api.deepseek.com",
            endpoints=[
                endpoint_for(self.name, "chat_completions", "/chat/completions"),
                # DeepSeek also exposes an Anthropic-format (Messages API) surface.
                endpoint_for(self.name, "messages", "/anthropic/v1/messages"),
            ],
            models=models,
        )

    def _fetch_model_ids(self) -> list[str]:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is required to scrape the DeepSeek models API")
        resp = self._client.get(
            MODELS_URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def _fetch_pricing(self) -> dict[str, dict]:
        try:
            resp = self._client.get(PRICING_URL, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        return _parse_pricing(resp.text)


def _parse_pricing(html: str) -> dict[str, dict]:
    """Parse the transposed pricing table into ``model id -> details`` dict."""
    tree = HTMLParser(html)
    table = tree.css_first("table")
    if table is None:
        return {}
    rows = [[c.text(strip=True) for c in tr.css("th,td")] for tr in table.css("tr")]
    if not rows or not rows[0] or rows[0][0].upper() != "MODEL":
        return {}

    model_ids = [_clean_id(c) for c in rows[0][1:]]
    n = len(model_ids)
    out: dict[str, dict] = {mid: {"capabilities": []} for mid in model_ids}
    # Per-model price components, filled as the relevant rows are found.
    inputs: list[float | None] = [None] * n
    outputs: list[float | None] = [None] * n
    cache: list[float | None] = [None] * n

    for row in rows[1:]:
        label, values = _label_values(row, n)
        if values is None:
            continue
        up = label.upper()
        if "CACHE HIT" in up:
            cache = [_money(v) for v in values]
        elif "CACHE MISS" in up or (up.startswith("1M INPUT") and "CACHE" not in up):
            inputs = [_money(v) for v in values]
        elif "OUTPUT TOKEN" in up:
            outputs = [_money(v) for v in values]
        elif up == "CONTEXT LENGTH":
            for mid, v in zip(model_ids, _broadcast(values, n)):
                out[mid]["context_window"] = _tokens(v)
        elif up == "MAX OUTPUT":
            for mid, v in zip(model_ids, _broadcast(values, n)):
                out[mid]["max_output_tokens"] = _tokens(v)
        elif up == "MODEL VERSION":
            for mid, v in zip(model_ids, _broadcast(values, n)):
                out[mid]["display_name"] = v
        elif all(v == "✓" for v in values) and values:
            cap = _slug(label)
            for mid, v in zip(model_ids, _broadcast(values, n)):
                if v == "✓" and cap:
                    out[mid]["capabilities"].append(cap)

    for i, mid in enumerate(model_ids):
        out[mid]["capabilities"] = sorted(out[mid]["capabilities"])
        if inputs[i] is not None or outputs[i] is not None:
            extra = {"cache_read": cache[i]} if cache[i] is not None else {}
            out[mid]["pricing"] = Pricing(input=inputs[i], output=outputs[i], extra=extra)
    return out


def _label_values(row: list[str], n: int) -> tuple[str, list[str] | None]:
    """Split a table row into (label, n per-model value cells).

    Rows may carry a leading section header cell (rowspan) so the label can span
    multiple leading cells; a single trailing value indicates a colspan that
    applies to every model.
    """
    if len(row) >= n + 1:
        return " ".join(row[:-n]), row[-n:]
    if len(row) == 2:  # single colspan value
        return row[0], [row[1]]
    return "", None


def _broadcast(values: list[str], n: int) -> list[str]:
    return values if len(values) == n else values * n


def _clean_id(text: str) -> str:
    return _FOOTNOTE.sub("", text)


def _slug(label: str) -> str:
    label = re.sub(r"[(（].*?[)）]", "", label)  # drop ascii and full-width parentheticals
    label = re.sub(r"^\s*features\b", "", label, flags=re.IGNORECASE)  # drop section prefix
    return "_".join(label.lower().split())


def _money(text: str) -> float | None:
    m = _MONEY.search(text or "")
    return float(m.group(1)) if m else None


def _tokens(text: str) -> int | None:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([MK])?", text or "")
    if not m:
        return None
    value = float(m.group(1))
    return int(value * {"M": 1_000_000, "K": 1000}.get(m.group(2), 1))

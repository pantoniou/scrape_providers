"""OpenCode Zen provider scraper — fully native (no OpenRouter).

Zen is the sst/opencode project's own gateway: one key, one base URL, models
from several vendors served over whichever wire protocol each model's vendor
speaks. Two native sources:

* ``GET https://opencode.ai/zen/v1/models`` (requires ``OPENCODE_API_KEY``) —
  the ids Zen actually serves right now.
* The Zen docs page (``opencode.ai/docs/zen``) — a models table (display name,
  model id, per-model endpoint) and a pricing table (input / output / cached
  read / cached write per million tokens). The pricing table is keyed by display
  name, so it is joined to ids through the models table.

Two wrinkles in the docs tables:

* Long-context models are priced in tiers, listed as separate rows suffixed
  ``(<= 200K tokens)`` / ``(> 200K tokens)``. Only the base tier is used, matching
  how the other providers' base prices are recorded.
* Free models list ``Free`` instead of a dollar figure; that is emitted as 0.0,
  not as missing pricing.

Zen exposes each model over its vendor's protocol (OpenAI ``responses``,
Anthropic ``messages``, Google ``generate_content``, plus an OpenAI-compatible
``chat/completions``), so all four surfaces are listed under ``endpoints``.
The docs give no context window, so models carry none unless another provider in
the catalog serves the same canonical model.
"""

from __future__ import annotations

import os
import re

import httpx
from selectolax.parser import HTMLParser

from ..base import Scraper
from ..models import Model, Pricing, Provider
from ..tools import endpoint_for

API_ROOT = "https://opencode.ai/zen/v1"
MODELS_URL = f"{API_ROOT}/models"
DOCS_URL = "https://opencode.ai/docs/zen"

_MONEY = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")
# " (<= 200K tokens)" / " (> 272K tokens)" context-tier suffix on pricing rows.
_TIER = re.compile(r"\s*\((?:[<>≤≥]|&[lg]t;)[^)]*\)\s*$")
# Pricing table column -> where it lands in Pricing.
_FIELDS = {"input": "input", "output": "output"}
_EXTRA = {"cached read": "cache_read", "cached write": "cache_write"}


class OpenCodeScraper(Scraper):
    name = "opencode"

    def scrape(self) -> Provider:
        docs = self._fetch_docs()  # model id -> {display_name, pricing}
        models = [
            Model(
                id=model_id,
                display_name=docs.get(model_id, {}).get("display_name"),
                modalities=["text"],
                pricing=docs.get(model_id, {}).get("pricing"),
            )
            for model_id in self._fetch_model_ids()
        ]
        return Provider(
            name=self.name,
            root_url=API_ROOT,
            endpoints=[
                endpoint_for(self.name, "chat_completions", "/chat/completions"),
                endpoint_for(self.name, "responses", "/responses"),
                endpoint_for(self.name, "messages", "/messages"),
                endpoint_for(self.name, "generate_content", "/models/{model}"),
            ],
            models=models,
        )

    def _fetch_model_ids(self) -> list[str]:
        key = os.environ.get("OPENCODE_API_KEY")
        if not key:
            raise RuntimeError("OPENCODE_API_KEY is required to scrape the OpenCode Zen models API")
        resp = self._client.get(
            MODELS_URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", []) if m.get("id")]

    def _fetch_docs(self) -> dict[str, dict]:
        try:
            resp = self._client.get(DOCS_URL, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        return _parse_docs(resp.text)


def _parse_docs(html: str) -> dict[str, dict]:
    """Join the docs' models table and pricing table into ``id -> details``."""
    tables = [_rows(t) for t in HTMLParser(html).css("table")]
    names = _models_table(tables)  # display name -> model id
    out: dict[str, dict] = {mid: {"display_name": name} for name, mid in names.items()}
    for name, pricing in _pricing_table(tables).items():
        model_id = names.get(name)
        if model_id:
            out[model_id]["pricing"] = pricing
    return out


def _rows(table) -> list[list[str]]:
    return [[c.text(strip=True) for c in tr.css("th,td")] for tr in table.css("tr")]


def _find(tables: list[list[list[str]]], *headers: str) -> tuple[list[str], list[list[str]]]:
    """Header + body rows of the first table whose header row starts with ``headers``."""
    for rows in tables:
        if rows and [h.strip().lower() for h in rows[0][: len(headers)]] == list(headers):
            return rows[0], rows[1:]
    return [], []


def _models_table(tables: list[list[list[str]]]) -> dict[str, str]:
    _, rows = _find(tables, "model", "model id")
    return {r[0]: r[1] for r in rows if len(r) >= 2 and r[0] and r[1]}


def _pricing_table(tables: list[list[list[str]]]) -> dict[str, Pricing]:
    header, rows = _find(tables, "model", "input")
    columns = [h.strip().lower() for h in header[1:]]
    out: dict[str, Pricing] = {}
    for row in rows:
        name = _TIER.sub("", row[0])
        # For tiered models keep the base (small-context) tier; a higher tier row
        # only fills in when its base tier is missing.
        base_tier = name == row[0] or _is_base_tier(row[0])
        if not name or (name in out and not base_tier):
            continue
        fields: dict[str, float] = {}
        extra: dict[str, float] = {}
        for column, cell in zip(columns, row[1:]):
            value = _money(cell)
            if value is None:
                continue
            if column in _FIELDS:
                fields[_FIELDS[column]] = value
            elif column in _EXTRA:
                extra[_EXTRA[column]] = value
        if fields:
            out[name] = Pricing(**fields, extra=extra)
    return out


def _is_base_tier(name: str) -> bool:
    """True for the ``(<= N tokens)`` row of a tiered model (vs ``(> N tokens)``)."""
    return bool(re.search(r"\((?:[<≤]|&lt;)", name))


def _money(text: str) -> float | None:
    if text.strip().lower() == "free":
        return 0.0
    m = _MONEY.search(text or "")
    return float(m.group(1)) if m else None

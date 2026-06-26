"""OpenAI provider scraper.

Model list and characteristics (context window, modalities, capabilities) come
from OpenRouter's ``openai/*`` entries — OpenAI's native ``GET /v1/models``
returns only bare ids.

**Pricing is scraped natively** from OpenAI's own pricing page
(``platform.openai.com/docs/pricing``) and joined onto the models by id, so the
figures are OpenAI's list prices rather than OpenRouter's routing rate. Models
not found on the page fall back to OpenRouter's pricing.
"""

from __future__ import annotations

import re

import httpx
from selectolax.parser import HTMLParser

from ..base import Scraper
from ..models import Pricing, Provider
from ..tools import endpoint_for
from . import openrouter

PRICING_URL = "https://platform.openai.com/docs/pricing"

# Header tokens that mark a table as something other than per-token model
# pricing (fine-tuning, media, tools, transcription); such tables are skipped.
_SKIP_COLUMNS = {"training", "size", "use case", "price per second", "tool", "estimated cost"}
_MODALITY_WORDS = {"text", "audio", "image", "video"}
_MONEY = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")


class OpenAIScraper(Scraper):
    name = "openai"
    _prefix = "openai/"

    def scrape(self) -> Provider:
        native_pricing = self._fetch_native_pricing()
        models = []
        for raw in openrouter.fetch_models(self._client):
            if not raw["id"].startswith(self._prefix):
                continue
            model = openrouter.to_model(raw, strip_prefix=True)
            override = native_pricing.get(model.id)
            if override is not None:
                model.pricing = override
            models.append(model)
        return Provider(
            name=self.name,
            root_url="https://api.openai.com",
            endpoints=[
                endpoint_for(self.name, "responses", "/v1/responses"),
                endpoint_for(self.name, "chat_completions", "/v1/chat/completions"),
            ],
            models=models,
        )

    def _fetch_native_pricing(self) -> dict[str, Pricing]:
        try:
            resp = self._client.get(PRICING_URL, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        return parse_openai_pricing(resp.text)


def parse_openai_pricing(html: str) -> dict[str, Pricing]:
    """Parse OpenAI's pricing page into ``model id -> Pricing`` (USD per MTok).

    Handles the several table shapes on the page (flagship two-tier
    Short/Long-context tables, ``Modality`` tables, and ``Category/Model``
    tables) by locating columns from their header names. The first table a model
    appears in wins, so standard pricing takes precedence over batch/flex tiers.
    """
    tree = HTMLParser(html)
    result: dict[str, Pricing] = {}
    for table in tree.css("table"):
        rows = [[c.text(strip=True) for c in tr.css("th,td")] for tr in table.css("tr")]
        header_idx = _find_header(rows)
        if header_idx is None:
            continue
        header = [h.lower() for h in rows[header_idx]]
        if _SKIP_COLUMNS & set(header):
            continue
        model_col = header.index("model")
        # First (standard / short-context) Input/Output and Cached input columns.
        input_col = _first_index(header, lambda h: h == "input")
        output_col = _first_index(header, lambda h: h.startswith("output"))
        cached_col = _first_index(header, lambda h: h == "cached input")
        if input_col is None or output_col is None:
            continue

        for row in rows[header_idx + 1 :]:
            if len(row) <= max(model_col, input_col, output_col):
                continue
            model_id = row[model_col]
            if not _looks_like_model(model_id) or model_id in result:
                continue
            pricing = _row_pricing(row, input_col, output_col, cached_col)
            if pricing is not None:
                result[model_id] = pricing
    return result


def _find_header(rows: list[list[str]]) -> int | None:
    """Index of the row that contains the real column header (has Model + Input).

    Flagship tables have a grouping row above the header, so this is not always
    row 0.
    """
    for i, row in enumerate(rows):
        lowered = [c.lower() for c in row]
        if "model" in lowered and "input" in lowered:
            return i
    return None


def _first_index(header: list[str], pred) -> int | None:
    for i, h in enumerate(header):
        if pred(h):
            return i
    return None


def _looks_like_model(cell: str) -> bool:
    cell = cell.strip()
    if not cell or cell.lower() in _MODALITY_WORDS:
        return False
    # Model ids are lowercase slugs; display rows like "Text"/"Audio" are not.
    return cell == cell.lower() and any(ch.isalpha() for ch in cell)


def _row_pricing(
    row: list[str], input_col: int, output_col: int, cached_col: int | None
) -> Pricing | None:
    input_price = _money(row[input_col])
    output_price = _money(row[output_col])
    if input_price is None and output_price is None:
        return None
    extra: dict[str, float] = {}
    if cached_col is not None and cached_col < len(row):
        cached = _money(row[cached_col])
        if cached is not None:
            extra["cache_read"] = cached
    return Pricing(input=input_price, output=output_price, extra=extra)


def _money(text: str) -> float | None:
    m = _MONEY.search(text or "")
    return float(m.group(1)) if m else None

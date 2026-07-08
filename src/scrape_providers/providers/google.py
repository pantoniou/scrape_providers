"""Google (Gemini) provider scraper.

Model list and characteristics (context window, modalities, capabilities) come
from OpenRouter's ``google/*`` entries — Google's native ``models.list`` API
requires an API key and doesn't expose pricing anyway.

**Pricing is scraped natively** from Google's own pricing page
(``ai.google.dev/gemini-api/docs/pricing``) and joined onto the models, with
OpenRouter pricing as fallback for models not on the page. Unlike OpenAI's
pricing page, Google's has no per-row model id: each model is an ``<h2>``
heading followed by one or more pricing tables (Standard/Batch/Flex/Priority
tiers under their own ``<h3>``), so the join key is a slug derived from the
heading text rather than a table column.
"""

from __future__ import annotations

import re

import httpx
from selectolax.parser import HTMLParser

from ..base import Scraper
from ..models import Pricing, Provider
from ..tools import endpoint_for
from . import openrouter

PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"

_MONEY = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")

# Image/media models are priced "per image" (or per second/frame) rather than
# per token; such figures must not be emitted as per_million_tokens.
_NON_TOKEN_UNIT = re.compile(r"per\s+(image|second|frame|minute|video)\b", re.IGNORECASE)

# Pricing-table row labels we care about, matched by prefix (case-insensitive).
_ROW_FIELDS = {
    "input price": "input",
    "output price": "output",
    "context caching price": "cache_read",
}

# Tier sub-headings (h3) that count as the headline rate. Batch/Flex/Priority
# are discounted/premium variants of the same Standard figures, so they're
# skipped; None means the model had no tier sub-heading at all.
_HEADLINE_TIERS = {None, "standard"}


class GoogleScraper(Scraper):
    name = "google"
    _prefix = "google/"

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
            root_url="https://generativelanguage.googleapis.com",
            endpoints=[
                endpoint_for(
                    self.name, "generate_content", "/v1beta/models/{model}:generateContent"
                ),
                endpoint_for(self.name, "chat_completions", "/v1beta/openai/chat/completions"),
            ],
            models=models,
        )

    def _fetch_native_pricing(self) -> dict[str, Pricing]:
        try:
            resp = self._client.get(PRICING_URL, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        return parse_gemini_pricing(resp.text)


def parse_gemini_pricing(html: str) -> dict[str, Pricing]:
    """Parse the pricing page into ``model slug -> Pricing`` (USD per MTok).

    Walks the document in order, tracking the current model (``<h2>``) and tier
    (``<h3>``); each ``<table>`` under a headline tier is parsed for its Input
    price / Output price / Context caching price rows, taking the figure from
    the "Paid Tier" column (the free tier is rate-limited, not a real price).
    """
    tree = HTMLParser(html)
    result: dict[str, Pricing] = {}
    slug: str | None = None
    tier: str | None = None
    for node in tree.root.traverse(include_text=False):
        if node.tag == "h2":
            slug = _slugify(node.text(strip=True))
            tier = None
        elif node.tag == "h3":
            tier = node.text(strip=True).lower()
        elif node.tag == "table" and slug and tier in _HEADLINE_TIERS:
            pricing = _table_pricing(node)
            if pricing is not None:
                result[slug] = pricing
    return result


def _table_pricing(table) -> Pricing | None:
    rows = [[c.text(strip=True) for c in tr.css("th,td")] for tr in table.css("tr")]
    if not rows:
        return None
    header = [h.lower() for h in rows[0]]
    paid_col = next((i for i, h in enumerate(header) if "paid" in h), len(header) - 1)
    fields: dict[str, float] = {}
    for row in rows[1:]:
        if paid_col >= len(row):
            continue
        label = row[0].lower()
        field = next((f for prefix, f in _ROW_FIELDS.items() if label.startswith(prefix)), None)
        if field is None:
            continue
        # Parentheticals are modality notes ("(text / image)"), not units.
        cell = re.sub(r"\(.*?\)", "", row[paid_col])
        if _NON_TOKEN_UNIT.search(cell):
            # Priced per image/second/etc., not per token ("$0.039 per image").
            # The whole table is unusable: emitting the other fields would
            # override the OpenRouter fallback's complete per-token pricing.
            return None
        value = _money(cell)
        if value is not None:
            fields[field] = value
    if "input" not in fields and "output" not in fields:
        return None
    extra = {k: v for k, v in fields.items() if k not in ("input", "output")}
    return Pricing(input=fields.get("input"), output=fields.get("output"), extra=extra)


def _money(text: str) -> float | None:
    m = _MONEY.search(text or "")
    return float(m.group(1)) if m else None


def _slugify(heading: str) -> str:
    """Turn a pricing-page heading into an id-shaped slug for joining to OpenRouter.

    Drops parenthetical nicknames ("(Nano Banana 2)") and emoji, then joins the
    remaining alphanumeric/dot tokens with hyphens, so "Gemini 3.1 Flash-Lite"
    and the OpenRouter id "gemini-3.1-flash-lite" normalize to the same slug.
    """
    heading = re.sub(r"\(.*?\)", "", heading)
    heading = re.sub(r"[^\x00-\x7F]+", "", heading)
    tokens = re.findall(r"[A-Za-z0-9.]+", heading)
    return "-".join(t.lower() for t in tokens)

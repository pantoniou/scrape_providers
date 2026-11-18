"""Fireworks AI provider scraper — fully native (no OpenRouter).

The authenticated models API supplies served IDs and capabilities. The public
serverless pricing table supplies published rates, including Fast routes.
Model pages and the public catalog supply additional per-model prices.
"""

from __future__ import annotations

import json
import os
import re

import httpx
from selectolax.parser import HTMLParser

from ..base import Scraper
from ..models import Model, Pricing, Provider
from ..tools import endpoint_for
from . import created_date

API_ROOT = "https://api.fireworks.ai/inference/v1"
MODELS_URL = f"{API_ROOT}/models"
PRICING_DOC = "https://docs.fireworks.ai/serverless/pricing.md"
MODEL_CATALOG = "https://fireworks.ai/models?modelTypes=Serverless&show=serverless"
MODEL_PAGE = "https://fireworks.ai/models/fireworks/{slug}"

_MONEY = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")
# The price block's label, e.g. "(input/cached input/output)".
_PRICE_LABEL = re.compile(r"\(([^)]*input[^)]*)\)")
# Which Pricing field each label component feeds.
_FIELDS = {"input": "input", "output": "output", "cached input": "cache_read"}


class FireworksScraper(Scraper):
    name = "fireworks"

    def scrape(self) -> Provider:
        models: list[Model] = []
        entries = self._fetch_models()
        prices = _parse_pricing_doc(self._fetch_public(PRICING_DOC))
        catalog_prices: dict[str, Pricing] | None = None
        for entry in entries:
            model_id = entry["id"]
            page = self._fetch_model_page(model_id)
            pricing = prices.get(model_id) or page.get("pricing")
            if pricing is None and model_id.startswith("accounts/fireworks/models/"):
                if catalog_prices is None:
                    catalog_prices = _parse_catalog_pricing(self._fetch_public(MODEL_CATALOG))
                pricing = catalog_prices.get(model_id)
            modalities = ["text"] + (["image"] if entry.get("supports_image_input") else [])
            models.append(
                Model(
                    id=model_id,
                    display_name=page.get("display_name"),
                    context_window=entry.get("context_length"),
                    modalities=modalities,
                    capabilities=["function_calling"] if entry.get("supports_tools") else [],
                    open_source=page.get("open_source", False),
                    introduced=created_date(entry),
                    pricing=pricing,
                )
            )
        return Provider(
            name=self.name,
            root_url=API_ROOT,
            endpoints=[
                endpoint_for(self.name, "chat_completions", "/chat/completions"),
                endpoint_for(self.name, "responses", "/responses"),
                # Fireworks also serves an Anthropic-format (Messages API) surface.
                endpoint_for(self.name, "messages", "/messages"),
            ],
            models=models,
        )

    def _fetch_models(self) -> list[dict]:
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError("FIREWORKS_API_KEY is required to scrape the Fireworks models API")
        resp = self._client.get(
            MODELS_URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}
        )
        resp.raise_for_status()
        return [m for m in resp.json().get("data", []) if m.get("id")]

    def _fetch_model_page(self, model_id: str) -> dict:
        """Best-effort scrape of a model's public page (display name, price, OSS)."""
        if "/models/" not in model_id:  # routers have no public page
            return {}
        html = self._fetch_public(MODEL_PAGE.format(slug=model_id.rsplit("/", 1)[-1]))
        return _parse_model_page(html) if html else {}

    def _fetch_public(self, url: str) -> str:
        try:
            resp = self._client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except httpx.HTTPError:
            return ""
        return resp.text


def _parse_pricing_doc(markdown: str) -> dict[str, Pricing]:
    """Read the Standard column, keeping alternate serving modes separate."""
    rows = re.findall(
        r"^\|\s*\[([^\]]+)\]\(https://app\.fireworks\.ai/models/fireworks/([\w.-]+)\)"
        r"\s*\|([^|\n]+)\|",
        markdown,
        re.MULTILINE,
    )
    prices = {}
    for name, slug, standard in rows:
        values = _MONEY.findall(standard)
        if len(values) != 3 or name.endswith("(US)") or " Ultra-fast " in name:
            continue
        model_id = (
            f"accounts/fireworks/routers/{slug}-fast"
            if name.endswith(" Fast")
            else f"accounts/fireworks/models/{slug}"
        )
        input_rate, cached, output = map(float, values)
        prices[model_id] = Pricing(input=input_rate, output=output, extra={"cache_read": cached})
    return prices


def _parse_catalog_pricing(html: str) -> dict[str, Pricing]:
    # The model library renders only the first page of cards. Its embedded
    # model list also contains the prices for cards behind "Load more".
    prices = {}
    decoder = json.JSONDecoder()
    for script in HTMLParser(html).css("script"):
        match = re.fullmatch(r"self\.__next_f\.push\((.*)\)", script.text(), re.DOTALL)
        if not match:
            continue
        try:
            record = json.loads(match[1])
        except ValueError:
            continue
        if len(record) < 2 or not isinstance(record[1], str):
            continue
        for marker in re.finditer(r'"models":(?=\[\{"_id":)', record[1]):
            try:
                models, _ = decoder.raw_decode(record[1], marker.end())
            except ValueError:
                continue
            for model in models:
                slug = model.get("slug", "")
                input_rate = model.get("pricingInput")
                output_rate = model.get("pricingOutput")
                if (
                    slug.startswith("fireworks/")
                    and model.get("pricingUnit") == "million"
                    and isinstance(input_rate, (int, float))
                    and isinstance(output_rate, (int, float))
                ):
                    prices[f"accounts/fireworks/models/{slug.removeprefix('fireworks/')}"] = (
                        Pricing(input=input_rate, output=output_rate)
                    )
    return prices


def _parse_model_page(html: str) -> dict:
    tree = HTMLParser(html)
    out: dict = {"open_source": "huggingface.co" in html}
    h1 = tree.css_first("h1")
    if h1 is not None:
        out["display_name"] = h1.text(strip=True)
    pricing = _parse_serverless_pricing(tree)
    if pricing is not None:
        out["pricing"] = pricing
    return out


def _parse_serverless_pricing(tree: HTMLParser) -> Pricing | None:
    """Read the "$in / $cached / $out per 1M tokens" block off a model page.

    The values and their label sit in sibling nodes, so the label node is located
    first (it names the order of the components) and the prices are read from the
    surrounding block. The label is the source of truth for the order: models
    without cached-input pricing publish only two components. Embedding/reranking
    models publish a single price (labelled or not), which is the input price.
    """
    # Ancestors of the label node also contain its text; the innermost match
    # (shortest text) is the label itself, whose parent holds label + prices.
    candidates = [
        node
        for node in tree.css("div")
        if "1M Tokens" in node.text(strip=True) and node.parent is not None
    ]
    for node in sorted(candidates, key=lambda n: len(n.text(strip=True))):
        label = _PRICE_LABEL.search(node.text(strip=True))
        values = [float(v) for v in _MONEY.findall(node.parent.text())]
        # A lone price is the input price, whether or not the label is the full
        # "(input/cached input/output)" one — embedding models keep that label
        # over a single figure, and there is no output to pair it with.
        if len(values) == 1:
            return Pricing(input=values[0])
        if label is None:
            continue
        components = [c.strip().lower() for c in label.group(1).split("/")]
        if len(values) != len(components):
            continue
        fields = {_FIELDS[c]: v for c, v in zip(components, values) if c in _FIELDS}
        extra = {"cache_read": fields.pop("cache_read")} if "cache_read" in fields else {}
        if not fields:
            continue
        return Pricing(**fields, extra=extra)
    return None

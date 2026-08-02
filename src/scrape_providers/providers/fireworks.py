"""Fireworks AI provider scraper — fully native (no OpenRouter).

Two native sources:

* ``GET https://api.fireworks.ai/inference/v1/models`` (requires
  ``FIREWORKS_API_KEY``) — the models Fireworks actually serves, with context
  length and tool/image support flags.
* One public model page per served model
  (``fireworks.ai/models/fireworks/<slug>``) — its "Available Serverless" block
  carries the per-million-token price triple (input / cached input / output) and
  the display name. Fireworks' ``/pricing`` page only covers fine-tuning and
  dedicated deployments, so serverless prices have to come from the model pages.

Fireworks' *routers* (``accounts/fireworks/routers/...``, the -fast/-turbo speed
tiers) have no public model page, so they are emitted without pricing rather
than inheriting the base model's — their whole point is a different price/speed
point.
"""

from __future__ import annotations

import os
import re

import httpx
from selectolax.parser import HTMLParser

from ..base import Scraper
from ..models import Model, Pricing, Provider
from ..tools import endpoint_for

API_ROOT = "https://api.fireworks.ai/inference/v1"
MODELS_URL = f"{API_ROOT}/models"
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
        for entry in self._fetch_models():
            model_id = entry["id"]
            page = self._fetch_model_page(model_id)
            modalities = ["text"] + (["image"] if entry.get("supports_image_input") else [])
            models.append(
                Model(
                    id=model_id,
                    display_name=page.get("display_name"),
                    context_window=entry.get("context_length"),
                    modalities=modalities,
                    capabilities=["function_calling"] if entry.get("supports_tools") else [],
                    open_source=page.get("open_source", False),
                    pricing=page.get("pricing"),
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
        try:
            resp = self._client.get(
                MODEL_PAGE.format(slug=model_id.rsplit("/", 1)[-1]),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        return _parse_model_page(resp.text)


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
    models publish a single unlabelled price, which is the input price.
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
        if label is None:
            if len(values) != 1:
                continue
            return Pricing(input=values[0])
        components = [c.strip().lower() for c in label.group(1).split("/")]
        if len(values) != len(components):
            continue
        fields = {_FIELDS[c]: v for c, v in zip(components, values) if c in _FIELDS}
        extra = {"cache_read": fields.pop("cache_read")} if "cache_read" in fields else {}
        if not fields:
            continue
        return Pricing(**fields, extra=extra)
    return None

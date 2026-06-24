"""OpenRouter provider scraper, and shared mapping for OpenRouter-sourced data.

OpenRouter exposes a single public, unauthenticated endpoint that lists every
model it routes to — including first-party OpenAI, DeepSeek, Anthropic, etc. —
with normalized pricing, context limits, modalities, and supported parameters.

This module is therefore the data source for several scrapers:

* :class:`OpenRouterScraper` emits the full catalog under the ``openrouter`` name.
* The OpenAI and DeepSeek scrapers reuse :func:`fetch_models` and
  :func:`to_model`, filtering by id prefix (their native APIs require keys and
  return only ids without pricing).

Pricing is OpenRouter's per-token rate converted to USD per million tokens; for
first-party models this tracks native list pricing but is not guaranteed to.
"""

from __future__ import annotations

import httpx

from ..base import Scraper
from ..models import Model, Pricing, Provider
from ..oss import is_open_source
from ..tools import endpoint_for

API_URL = "https://openrouter.ai/api/v1/models"

# OpenRouter pricing keys (USD per token) -> our fields. prompt/completion are
# the headline input/output; everything else is carried in Pricing.extra.
_PRICE_FIELDS = {
    "prompt": "input",
    "completion": "output",
    "input_cache_read": "cache_read",
    "input_cache_write": "cache_write",
    "web_search": "web_search",
    "internal_reasoning": "reasoning",
    "request": "per_request",
    "image": "image",
}


def fetch_models(client: httpx.Client) -> list[dict]:
    resp = client.get(API_URL)
    resp.raise_for_status()
    return resp.json().get("data", [])


def to_model(raw: dict, *, strip_prefix: bool = False) -> Model:
    full_id = raw["id"]
    model_id = full_id.split("/", 1)[1] if strip_prefix and "/" in full_id else full_id
    arch = raw.get("architecture") or {}
    top = raw.get("top_provider") or {}
    return Model(
        id=model_id,
        display_name=raw.get("name"),
        context_window=raw.get("context_length") or top.get("context_length"),
        max_output_tokens=top.get("max_completion_tokens"),
        modalities=list(arch.get("input_modalities") or []),
        capabilities=sorted(raw.get("supported_parameters") or []),
        open_source=is_open_source(full_id),  # detect from the prefixed id
        pricing=_pricing(raw.get("pricing")),
    )


def _pricing(raw: dict | None) -> Pricing | None:
    if not raw:
        return None
    fields: dict[str, float] = {}
    for source, field in _PRICE_FIELDS.items():
        value = _per_million(raw.get(source))
        if value is not None:
            fields[field] = value
    if not fields:
        return None
    extra = {k: v for k, v in fields.items() if k not in ("input", "output")}
    return Pricing(input=fields.get("input"), output=fields.get("output"), extra=extra)


def _per_million(value: object) -> float | None:
    """OpenRouter quotes USD per token as a string; convert to per-million."""
    if value in (None, ""):
        return None
    try:
        per_token = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if per_token < 0:
        return None  # OpenRouter uses negative values as a not-applicable sentinel
    # 'request' and similar flat fees aren't per-token, but OpenRouter still
    # quotes them per unit; scale uniformly and let the field name disambiguate.
    return round(per_token * 1_000_000, 6)


class OpenRouterScraper(Scraper):
    name = "openrouter"

    def scrape(self) -> Provider:
        models = [to_model(raw) for raw in fetch_models(self._client)]
        return Provider(
            name=self.name,
            root_url="https://openrouter.ai",
            endpoints=[
                endpoint_for(self.name, "chat_completions", "/api/v1/chat/completions")
            ],
            models=models,
        )

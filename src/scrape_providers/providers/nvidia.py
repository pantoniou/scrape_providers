"""NVIDIA Build (build.nvidia.com / NIM API catalog) provider scraper.

NVIDIA's hosted API is OpenAI-compatible at ``https://integrate.api.nvidia.com/v1``.
Sources:

* ``GET /v1/models`` — the ids actually served. The listing is public;
  ``NVIDIA_API_KEY`` (an ``nvapi-…`` key from build.nvidia.com/settings/api-keys)
  is optional and sent when present. It carries no context window, modalities or
  prices, and its ``created`` is a constant placeholder, so it is ignored.
* **models.dev** (``api.json`` → ``nvidia.models``) — display name, context /
  output limits, modalities, capability flags, open-weights flag, release date and
  cost. NVIDIA publishes none of this in a machine-readable form.

Prices are 0.0: the hosted catalog is free for prototyping (rate-limited per key,
not billed per token), which models.dev records as ``cost: {input: 0, output: 0}``.
Ids models.dev has not catalogued yet are emitted with no metadata but the same free price. Non-chat models
(image/video/speech/transcription generation) don't fit the per-million-token
shape and are dropped; embedding and reranking models are kept, as for Fireworks.
"""

from __future__ import annotations

import os
from datetime import date

import httpx

from ..base import Scraper
from ..models import Model, Pricing, Provider
from ..tools import endpoint_for

API_ROOT = "https://integrate.api.nvidia.com/v1"
MODELS_URL = f"{API_ROOT}/models"
MODELS_DEV_URL = "https://models.dev/api.json"

# models.dev `type` values that are not priced per token.
_SKIPPED_TYPES = {"image", "video", "speech", "transcription"}


class NvidiaScraper(Scraper):
    name = "nvidia"

    def scrape(self) -> Provider:
        meta = self._fetch_models_dev()
        models = []
        for model_id in self._fetch_model_ids():
            entry = meta.get(model_id, {})
            if entry.get("type") in _SKIPPED_TYPES:
                continue
            models.append(_model(model_id, entry))
        return Provider(
            name=self.name,
            root_url=API_ROOT,
            endpoints=[endpoint_for(self.name, "chat_completions", "/chat/completions")],
            models=models,
        )

    def _fetch_model_ids(self) -> list[str]:
        headers = {"Accept": "application/json"}
        key = os.environ.get("NVIDIA_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = self._client.get(MODELS_URL, headers=headers)
        if resp.status_code in (401, 403) and key:  # revoked key; the listing is public
            resp = self._client.get(MODELS_URL, headers={"Accept": "application/json"})
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", []) if m.get("id")]

    def _fetch_models_dev(self) -> dict[str, dict]:
        try:
            resp = self._client.get(MODELS_DEV_URL, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            return resp.json()["nvidia"]["models"]
        except (httpx.HTTPError, KeyError, ValueError):
            return {}


def _model(model_id: str, entry: dict) -> Model:
    modalities = (entry.get("modalities") or {}).get("input") or ["text"]
    limit = entry.get("limit") or {}
    caps = [
        cap
        for flag, cap in (
            ("tool_call", "function_calling"),
            ("structured_output", "structured_outputs"),
            ("reasoning", "reasoning"),
        )
        if entry.get(flag)
    ]
    cost = entry.get("cost") or {}
    pricing = (
        Pricing(input=cost["input"], output=cost.get("output"))
        if isinstance(cost.get("input"), (int, float))
        # models.dev lags the live listing (about 25 of 84 ids matched), but the
        # whole hosted catalog is free, so an uncatalogued id is free too.
        else Pricing(input=0.0, output=0.0)
    )
    return Model(
        id=model_id,
        display_name=entry.get("name"),
        context_window=limit.get("context") or None,
        max_output_tokens=limit.get("output") or None,
        modalities=list(modalities),
        capabilities=caps,
        open_source=bool(entry.get("open_weights")),
        introduced=_release_date(entry.get("release_date")),
        pricing=pricing,
    )


def _release_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None

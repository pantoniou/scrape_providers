"""Curated model allowlist to cut catalog noise.

Providers expose dozens to hundreds of models (old generations, size variants,
previews). The CLI's ``--curated`` flag filters each provider down to the ids
listed here: the current flagship/latest models only, plus a hand-picked set of
the latest open-source flagships from OpenRouter (GLM, Qwen, Llama, Mistral,
Kimi).

This is an explicit, manually maintained list — bump the ids when a new
generation ships. Ids that aren't present in a live scrape are simply skipped.

The built-in defaults below can be overridden persistently by a config file
(``--set-curated``); :func:`load_curated` returns the config if present, else the
defaults. The config path is ``$SCRAPE_PROVIDERS_CURATED`` or
``$XDG_CONFIG_HOME/scrape-providers/curated.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .models import Provider

# Provider name -> curated model ids (as they appear after each scraper's
# normalization, i.e. OpenAI/DeepSeek ids have the OpenRouter prefix stripped,
# OpenRouter open-source ids keep their vendor prefix).
DEFAULT_CURATED: dict[str, list[str]] = {
    "anthropic": [
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "claude-fable-5",
    ],
    "openai": [
        "gpt-5.4-nano",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-terra-pro",
        "gpt-5.6-luna",
        "gpt-5.6-luna-pro",
    ],
    "deepseek": [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ],
    "google": [
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
    ],
    # Latest open-weight flagships routed via OpenRouter (truly open weights —
    # Qwen's -max / Mistral's -large tiers are closed, so the open variants are
    # picked instead).
    "openrouter": [
        "z-ai/glm-5.2",
        "qwen/qwen3.5-397b-a17b",
        # llama-4-maverick dropped: superseded in its price band ($0.15/$0.6) by
        # the newer mistral-small-2603 (Apr 2025 vs Mar 2026).
        "mistralai/mistral-small-2603",
        "moonshotai/kimi-k2.7-code",
    ],
}


# Backwards-compatible alias for the built-in defaults.
CURATED = DEFAULT_CURATED


def config_path() -> Path:
    env = os.environ.get("SCRAPE_PROVIDERS_CURATED")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "scrape-providers" / "curated.yaml"


def load_curated() -> dict[str, list[str]]:
    """Return the active curated mapping: the config file if present, else defaults."""
    path = config_path()
    if path.exists():
        data = yaml.safe_load(path.read_text("utf-8")) or {}
        validate_curated(data)
        return data
    return DEFAULT_CURATED


def validate_curated(data: object) -> None:
    """Raise ValueError unless ``data`` is a mapping of provider -> list of str ids."""
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, list) and all(isinstance(i, str) for i in v)
        for k, v in data.items()
    ):
        raise ValueError("curated list must be a mapping of provider name -> list of model ids")


def save_curated(mapping: dict[str, list[str]]) -> Path:
    """Validate and persist ``mapping`` as the curated config; return its path."""
    validate_curated(mapping)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(mapping, sort_keys=True), "utf-8")
    return path


def curate(provider: Provider) -> Provider:
    """Return a copy of ``provider`` keeping only its curated models.

    Models are ordered to match the curated list. Providers with no curated
    entry are returned with an empty model list.
    """
    allowed = load_curated().get(provider.name, [])
    by_id = {m.id: m for m in provider.models}
    models = [by_id[i] for i in allowed if i in by_id]
    return provider.model_copy(update={"models": models})

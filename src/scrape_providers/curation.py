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
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-fable-5",
    ],
    "openai": [
        "gpt-oss-20b:free",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-luna-pro",
        "gpt-5.6-terra",
        "gpt-5.6-terra-pro",
        "gpt-5.6-sol",
        "gpt-5.6-sol-pro",
    ],
    "deepseek": [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ],
    # Fireworks ids are full resource paths; the routers (-fast/-turbo speed
    # tiers) are left out since they carry no published pricing.
    "fireworks": [
        "accounts/fireworks/models/kimi-k3",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/deepseek-v4-flash",
        "accounts/fireworks/models/qwen3p7-plus",
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/inkling",
    ],
    "opencode": [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gemini-3.6-flash",
        "gemini-3.1-pro",
        "grok-4.5",
        "kimi-k3",
        "glm-5.2",
        "minimax-m3",
        "deepseek-v4-pro",
    ],
    "google": [
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
    ],
    # Latest flagships routed via OpenRouter, open- and closed-weight alike:
    # OpenRouter is a real second (or third) route to the same model, with its
    # own pricing, so the models curated under their native providers above are
    # listed here too. Among the open-weight families the truly open variants are
    # picked (Qwen's -max / Mistral's -large tiers are closed).
    "openrouter": [
        "anthropic/claude-fable-5",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
        "cohere/command-a",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
        "google/gemini-3.1-flash-lite",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.5-flash",
        "google/gemini-3.6-flash",
        "~google/gemini-flash-latest",
        "~google/gemini-pro-latest",
        "google/gemma-4-26b-a4b-it",
        "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-31b-it",
        "google/gemma-4-31b-it:free",
        "meituan/longcat-2.0",
        "meta/muse-spark-1.1",
        "microsoft/phi-4",
        "minimax/minimax-m3",
        "mistralai/mistral-small-2603",
        "moonshotai/kimi-k2.7-code",
        "moonshotai/kimi-k3",
        "~moonshotai/kimi-latest",
        "nousresearch/hermes-4-405b",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.5",
        "openai/gpt-5.6-luna",
        "openai/gpt-5.6-luna-pro",
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-sol-pro",
        "openai/gpt-5.6-terra",
        "openai/gpt-5.6-terra-pro",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b:free",
        "openrouter/auto",
        "openrouter/auto-beta",
        "openrouter/bodybuilder",
        "openrouter/free",
        "openrouter/fusion",
        "openrouter/pareto-code",
        "poolside/laguna-s-2.1",
        "poolside/laguna-s-2.1:free",
        "qwen/qwen3.5-397b-a17b",
        "qwen/qwen3.7-plus",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.6-35b-a3b",
        "qwen/qwen3-8b",
        "tencent/hy3",
        "thinkingmachines/inkling",
        "x-ai/grok-4.5",
        "~x-ai/grok-latest",
        "xiaomi/mimo-v2.5",
        "z-ai/glm-5.2",
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

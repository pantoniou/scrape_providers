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
import sys
from pathlib import Path

import yaml

from .models import Provider

# Provider name -> curated model ids (as they appear after each scraper's
# normalization, i.e. OpenAI/DeepSeek ids have the OpenRouter prefix stripped,
# OpenRouter open-source ids keep their vendor prefix).
DEFAULT_CURATED: dict[str, list[str]] = {
    "anthropic": [
        "claude-fable-5-1",
        "claude-haiku-5-5",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
    ],
    "deepseek": [
        "deepseek-flash",
        "deepseek-v4-pro",
    ],
    # Fireworks ids are full resource paths; the routers (-fast/-turbo speed
    # tiers) are left out since they carry no published pricing.
    "fireworks": [
        "accounts/fireworks/models/deepseek-v4p1-flash",
        "accounts/fireworks/models/ember-1",
        "accounts/fireworks/models/glm-5p3",
        "accounts/fireworks/models/glm-5p3-flash",
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/inkling",
        "accounts/fireworks/models/kimi-k3",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/nemotron-3-ultra-nvfp4",
        "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
        "accounts/fireworks/models/qwen3-embedding-8b",
        "accounts/fireworks/models/qwen3p8-2p4t-a95b",
        "accounts/fireworks/models/qwen3p8-max",
        "accounts/fireworks/models/qwen3-reranker-8b",
        "accounts/fireworks/routers/glm-5p3-fast",
        "accounts/fireworks/routers/kimi-k3-fast",
        "firerouter/auto",
    ],
    "google": [
        "gemini-3.8-flash",
        "gemini-3.8-flash:batch",
        "gemma-4-26b-a4b-it",
        "gemma-4-26b-a4b-it:free",
        "gemma-4-31b-it",
        "gemma-4-31b-it:free",
    ],
    "meta": [
        "muse-spark-1.3",
        "muse-spark-1.3-contributor",
    ],
    "openai": [
        "gpt-5.4-mini",
        "gpt-6.1-sol",
        "gpt-6-astra",
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-oss-120b",
        "gpt-oss-20b",
    ],
    "opencode": [
        "big-pickle",
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-fable-5-1",
        "deepseek-v4-flash",
        "deepseek-v4-flash-free",
        "deepseek-v4-pro",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gemini-3.8-flash",
        "gemini-3.1-pro",
        "grok-4.7",
        "jev-1.13",
	    "jev-1.13-free",
        "kimi-k3",
        "glm-5.2",
        "minimax-m3",
        "muse-spark-1.3",
        "muse-spark-1.3-contributor-free",
    ],
    # Latest flagships routed via OpenRouter, open- and closed-weight alike:
    # OpenRouter is a real second (or third) route to the same model, with its
    # own pricing, so the models curated under their native providers above are
    # listed here too. Among the open-weight families the truly open variants are
    # picked (Qwen's -max / Mistral's -large tiers are closed).
    "openrouter": [
        "anthropic/claude-fable-5.1",
        "~anthropic/claude-fable-latest",
        "anthropic/claude-haiku-5.5",
        "~anthropic/claude-haiku-latest",
        "anthropic/claude-opus-5.5",
        "~anthropic/claude-opus-latest",
        "anthropic/claude-sonnet-5.5",
        "~anthropic/claude-sonnet-latest",
        "bytedance-seed/seed-2-1-turbo",
        "cognitivecomputations/dolphin-mistral-24b-venice-edition",
        "cohere/command-a",
        "deepseek/deepseek-v4-flash",
        "~deepseek/deepseek-flash-latest",
        "deepseek/deepseek-v4-pro",
        "~deepseek/deepseek-pro-latest",
        "dots-studio/dots-3-note-preview:free",
        "fireworks/ember-1",
        "google/gemini-3.8-flash",
        "~google/gemini-flash-latest",
        "google/gemma-4-31b-it",
        "google/gemma-4-31b-it:free",
        "gryphe/mythomax-l2-13b",
        "inception/mercury-2.5",
        "inclusionai/ling-3.1-flash",
        "inference-net/schematron-v2-turbo",
        "liquid/lfm-2.5-2.6b:free",
        "meituan/longcat-2.0",
        "meta/muse-spark-1.3",
        "meta/muse-spark-1.3-contributor",
        "microsoft/phi-4",
        "minimax/minimax-m3",
        "mistralai/mistral-large-4-0",
        "moonshotai/kimi-k3",
        "~moonshotai/kimi-latest",
        "nvidia/nemotron-3.5-content-safety",
        "nvidia/nemotron-3.5-content-safety:free",
        "nvidia/nemotron-3.5-lightning",
        "nvidia/nemotron-3.5-lightning:free",
        "nvidia/switchyard",
        "openai/gpt-6.1-sol",
        "openai/gpt-6-astra",
        "openai/gpt-6-luna",
        "openai/gpt-6-sol",
        "~openai/gpt-astra-latest",
        "~openai/gpt-luna-latest",
        "~openai/gpt-sol-latest",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-120b:batch",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-20b:batch",
        "openai/gpt-oss-safeguard-20b",
        "openrouter/auto",
        "openrouter/auto-beta",
        "openrouter/bodybuilder",
        "openrouter/free",
        "openrouter/fusion",
        "openrouter/pareto-code",
        "poolside/laguna-s-2.1",
        "poolside/laguna-s-2.1:free",
        "prism-ml/ternary-bonsai-2-27b",
        "qwen/qwen3.8-2.4t-a95b",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.8-flash",
        "stepfun/step-5-preview",
        "tencent/hy3",
        "tencent/hy4-preview",
        "thinkingmachines/inkling",
        "thinkingmachines/inkling:free",
        "thinkingmachines/inkling-small",
        "thinkingmachines/inkling-small:free",
        "typesafe/jev-router",
        "unbiased/pareto",
        "x-ai/grok-4.7",
        "~x-ai/grok-latest",
        "xiaomi/mimo-v2.6-flash",
        "xiaomi/mimo-v2.6-pro",
        "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash",
        "~z-ai/glm-flash-latest",
        "~z-ai/glm-latest",
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

    A curated id the provider doesn't serve is dropped, but reported on stderr:
    silence here reads exactly like the model being absent from the catalog on
    purpose, when the real cause is usually a typo (`gemini-flash-3.8` for
    `gemini-3.8-flash`) or a vendor renaming an id out from under the list.
    """
    allowed = load_curated().get(provider.name, [])
    by_id = {m.id: m for m in provider.models}
    missing = [i for i in allowed if i not in by_id]
    if missing:
        print(
            f"warning: {provider.name}: {len(missing)} curated model(s) not served: "
            f"{', '.join(missing)}",
            file=sys.stderr,
        )
    models = [by_id[i] for i in allowed if i in by_id]
    return provider.model_copy(update={"models": models})

"""Canonical model identity across providers.

The same logical model is named differently by each provider (OpenAI's
``gpt-5.5`` vs OpenRouter's ``openai/gpt-5.5``; DeepSeek's ``deepseek-v4-pro`` vs
OpenRouter's ``deepseek/deepseek-v4-pro``). :func:`canonical_id` maps a
provider-specific id to a shared key so capabilities can be stored once under
``models`` and referenced by every provider that serves the model.

The default rule drops any vendor prefix, lowercases, and restores the decimal
point in Fireworks-style version numbers (``kimi-k2p6`` -> ``kimi-k2.6``, so it
collapses with everyone else's ``kimi-k2.6``). ``ALIASES`` is an explicit
override table for ids the rule would get wrong.
"""

from __future__ import annotations

import re

# provider-specific model id -> canonical id (for cases the rule mishandles).
ALIASES: dict[str, str] = {}

# Fireworks writes version decimals as "p" (`glm-5p2`, `qwen3p7-plus`); no other
# provider's ids contain a digit-p-digit run, so undoing it is unambiguous.
_VERSION_P = re.compile(r"(?<=\d)p(?=\d)")


def canonical_id(model_id: str) -> str:
    if model_id in ALIASES:
        return ALIASES[model_id]
    return _VERSION_P.sub(".", model_id.split("/")[-1].strip().lower())

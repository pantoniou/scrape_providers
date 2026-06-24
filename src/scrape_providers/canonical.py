"""Canonical model identity across providers.

The same logical model is named differently by each provider (OpenAI's
``gpt-5.5`` vs OpenRouter's ``openai/gpt-5.5``; DeepSeek's ``deepseek-v4-pro`` vs
OpenRouter's ``deepseek/deepseek-v4-pro``). :func:`canonical_id` maps a
provider-specific id to a shared key so capabilities can be stored once under
``models`` and referenced by every provider that serves the model.

The default rule drops any vendor prefix and lowercases. ``ALIASES`` is an
explicit override table for ids the rule would get wrong.
"""

from __future__ import annotations

# provider-specific model id -> canonical id (for cases the rule mishandles).
ALIASES: dict[str, str] = {}


def canonical_id(model_id: str) -> str:
    if model_id in ALIASES:
        return ALIASES[model_id]
    return model_id.split("/")[-1].strip().lower()

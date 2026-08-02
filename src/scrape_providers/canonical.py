"""Canonical model identity across providers.

The same logical model is named differently by each provider (OpenAI's
``gpt-5.5`` vs OpenRouter's ``openai/gpt-5.5``; DeepSeek's ``deepseek-v4-pro`` vs
OpenRouter's ``deepseek/deepseek-v4-pro``). :func:`canonical_id` maps a
provider-specific id to a shared key so capabilities can be stored once under
``models`` and referenced by every provider that serves the model.

The default rule drops any vendor prefix, lowercases, and normalizes the two
ways providers mangle a version number:

* Fireworks writes the decimal point as ``p`` (``kimi-k2p6`` -> ``kimi-k2.6``).
* Anthropic writes it as a dash and may append a release date
  (``claude-opus-4-5-20251101`` -> ``claude-opus-4.5``), where OpenRouter and
  OpenCode Zen both use ``claude-opus-4.5`` / ``claude-opus-4-5``.

``ALIASES`` is an explicit override table for ids the rules would get wrong.
"""

from __future__ import annotations

import re

# provider-specific model id -> canonical id (for cases the rule mishandles).
ALIASES: dict[str, str] = {}

# Fireworks writes version decimals as "p" (`glm-5p2`, `qwen3p7-plus`); no other
# provider's ids contain a digit-p-digit run, so undoing it is unambiguous.
_VERSION_P = re.compile(r"(?<=\d)p(?=\d)")

# Anthropic-only: `claude-opus-4-8` is OpenRouter's `claude-opus-4.8`, and the
# native API also serves dated snapshot ids for older generations. Both rewrites
# are scoped to `claude-` ids so no other provider's dashes or digits are touched.
_CLAUDE = re.compile(r"^claude-")
_CLAUDE_DATE = re.compile(r"-\d{8}$")
_CLAUDE_VERSION = re.compile(r"(?<=\d)-(?=\d)")


def canonical_id(model_id: str) -> str:
    if model_id in ALIASES:
        return ALIASES[model_id]
    name = _VERSION_P.sub(".", model_id.split("/")[-1].strip().lower())
    if _CLAUDE.match(name):
        name = _CLAUDE_VERSION.sub(".", _CLAUDE_DATE.sub("", name))
    return name

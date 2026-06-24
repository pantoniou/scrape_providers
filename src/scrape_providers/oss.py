"""Open-source / open-weight model detection.

There is no single authoritative flag across providers, so this is a curated
heuristic: a model is treated as open-source if it comes from a vendor that
releases open weights, or its name matches a known open-weight family
(``gpt-oss``, ``gemma``, ``phi``) even when the vendor also ships closed models.
"""

from __future__ import annotations

# OpenRouter vendor prefixes that release open weights for (effectively) all
# their listed models.
OPEN_WEIGHT_VENDORS = {
    "meta-llama",
    "mistralai",
    "qwen",
    "z-ai",
    "moonshotai",
    "deepseek",
    "nvidia",
    "nousresearch",
    "allenai",
    "arcee-ai",
    "cognitivecomputations",
    "thudm",
    "liquid",
}

# Open-weight families from vendors that also ship proprietary models.
OPEN_WEIGHT_NAME_MARKERS = ("gpt-oss", "gemma", "phi")

# Closed, API-only families from otherwise-open vendors (Qwen's max/plus tiers,
# Mistral's large/medium). These override the open-vendor default.
CLOSED_NAME_MARKERS = ("-max", "-plus", "mistral-large", "mistral-medium")


def is_open_source(model_id: str) -> bool:
    vendor, _, name = model_id.partition("/")
    if not name:  # no vendor prefix; treat whole id as the name
        name = vendor
    name = name.lower()
    if any(marker in name for marker in OPEN_WEIGHT_NAME_MARKERS):
        return True
    if any(marker in name for marker in CLOSED_NAME_MARKERS):
        return False
    return vendor in OPEN_WEIGHT_VENDORS

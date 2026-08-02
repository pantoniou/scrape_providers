"""Controlled vocabulary for model modalities and capabilities.

Every provider names the same things differently, and some don't name
capabilities at all — they list the *request parameters* their API accepts.
OpenRouter (and the `openai`/`google` scrapers built on it) reports `top_a`,
`min_p` and `logit_bias` alongside `tools` and `reasoning`; Anthropic reports
product features (`citations`, `batch`); DeepSeek reports a mixture
(`tool_calls`, but also `anthropic_api`, which is a wire protocol). Comparing
models across providers means first agreeing on what a capability *is*.

Two rules draw the line:

* A **capability** is something the model can do that changes what you can build
  with it. A knob that tunes an ability every model already has is not one:
  `temperature` and `top_k` shape sampling, `max_tokens` bounds a response,
  `tool_choice` steers tool use that `tool_calling` already covers.
* A capability is a property of the **model**, not of the route to it.
  DeepSeek's `anthropic_api` describes a wire protocol and belongs on the
  provider's endpoint, where the catalog already records it.

The vocabulary is applied only to the merged, vendor-authoritative entries under
top-level ``models``. Each offering's ``reported_*`` fields keep the provider's
own words verbatim — normalizing those would destroy the evidence they exist to
preserve.

Terms in neither table are dropped with a warning, so a provider's new
vocabulary surfaces as something to classify rather than silently becoming a
capability or silently disappearing.
"""

from __future__ import annotations

# Canonical modalities. A flat set of what the model accepts; output is text for
# every model in the catalog, so there is nothing yet to distinguish.
MODALITIES: tuple[str, ...] = ("text", "image", "audio", "video", "pdf")

# Canonical capabilities, in emission order (grouped by kind, not alphabetized).
CAPABILITIES: tuple[str, ...] = (
    "tool_calling",
    "parallel_tool_calls",
    "structured_outputs",
    "json_mode",
    "reasoning",
    "reasoning_effort",
    "logprobs",
    "predicted_outputs",
    "prefix_completion",
    "web_search",
    "code_execution",
    "citations",
    "context_management",
    "batch",
)

# provider term -> canonical modality. "file" is OpenRouter's, Google's and
# OpenAI's word for document input, which in practice means PDF — the same thing
# Anthropic calls "pdf".
_MODALITY_ALIASES: dict[str, str] = {
    "file": "pdf",
    "document": "pdf",
    "image_input": "image",
    "audio_input": "audio",
}

# provider term -> canonical capability.
_CAPABILITY_ALIASES: dict[str, str] = {
    # Tool use. OpenRouter names the request field (`tools`), DeepSeek the
    # response shape (`tool_calls`), Fireworks the feature (`function_calling`).
    "tools": "tool_calling",
    "tool_calls": "tool_calling",
    "function_calling": "tool_calling",
    "parallel_tool_calls": "parallel_tool_calls",
    # Constrained output. A caller-supplied schema is a strictly stronger
    # guarantee than "valid JSON", so the two stay separate.
    "structured_outputs": "structured_outputs",
    "response_format": "json_mode",
    "json_output": "json_mode",
    # Reasoning. `include_reasoning` is OpenRouter's flag for returning the
    # trace, which only models that reason accept; `thinking` is Anthropic's name
    # for the same ability.
    "reasoning": "reasoning",
    "include_reasoning": "reasoning",
    "thinking": "reasoning",
    "reasoning_effort": "reasoning_effort",
    "effort": "reasoning_effort",
    # Token probabilities: `top_logprobs` is meaningless without `logprobs`, so
    # both denote the one ability.
    "logprobs": "logprobs",
    "top_logprobs": "logprobs",
    "prediction": "predicted_outputs",
    "chat_prefix_completion": "prefix_completion",
    "web_search_options": "web_search",
    "web_search": "web_search",
    "code_execution": "code_execution",
    "citations": "citations",
    "context_management": "context_management",
    "batch": "batch",
}

# Terms deliberately dropped, so that an unclassified term can be told apart
# from one already judged not to be a capability.
_IGNORED: frozenset[str] = frozenset(
    {
        # Sampling and decoding knobs: they tune generation, which every model
        # does, rather than adding an ability.
        "temperature",
        "top_p",
        "top_k",
        "top_a",
        "min_p",
        "seed",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "logit_bias",
        "verbosity",
        # Response-length bounds.
        "max_tokens",
        "max_completion_tokens",
        # Steering of tool use that `tool_calling` already implies.
        "tool_choice",
        # Modalities Anthropic also lists as features; already carried as
        # modalities, where they belong.
        "image_input",
        "pdf_input",
        # A wire protocol, not a model capability: the catalog records which
        # protocols a provider speaks under its `endpoints`.
        "anthropic_api",
    }
)


def _normalize(
    terms: list[str],
    aliases: dict[str, str],
    valid: tuple[str, ...],
    ignored: frozenset[str],
) -> tuple[list[str], list[str]]:
    """Map provider terms to the canonical vocabulary.

    Returns the canonical terms in ``valid``'s declared order (deduplicated, so
    two provider words for one ability collapse) and any unrecognized terms.
    """
    seen: set[str] = set()
    unknown: list[str] = []
    for term in terms:
        key = term.strip().lower()
        if not key or key in ignored:
            continue
        canonical = aliases.get(key, key)
        if canonical in valid:
            seen.add(canonical)
        elif key not in unknown:
            unknown.append(key)
    return [t for t in valid if t in seen], unknown


def normalize_modalities(terms: list[str]) -> tuple[list[str], list[str]]:
    """Canonical modalities plus any unrecognized terms."""
    return _normalize(terms, _MODALITY_ALIASES, MODALITIES, frozenset())


def normalize_capabilities(terms: list[str]) -> tuple[list[str], list[str]]:
    """Canonical capabilities plus any unrecognized terms."""
    return _normalize(terms, _CAPABILITY_ALIASES, CAPABILITIES, _IGNORED)

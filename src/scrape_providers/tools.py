"""Curated registry of built-in tools each provider exposes, by execution site.

These tools are not enumerated by any provider API — they live only in docs — so
this is a hand-maintained map keyed by ``(provider, protocol)``. Tools split by
where they run:

* ``hosted`` — executed on the provider's servers; the caller just gets results
  (web search, code interpreter, file search, image generation, …).
* ``local`` — the model emits an action and the **caller** runs it in their own
  environment (the bash/``local_shell`` tool, computer use, function calling, …).

The same model exposes different tools under different protocols (e.g. OpenAI's
``responses`` has ``local_shell`` while plain ``chat_completions`` does not), which
is why tools are attached to a provider's endpoints rather than to the model.
"""

from __future__ import annotations

from .models import Endpoint

# (provider, protocol) -> {"hosted": [...], "local": [...]}
TOOLS: dict[tuple[str, str], dict[str, list[str]]] = {
    ("openai", "responses"): {
        "hosted": ["web_search", "file_search", "code_interpreter", "image_generation"],
        "local": ["local_shell", "computer_use", "function_calling", "mcp"],
    },
    ("openai", "chat_completions"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    ("anthropic", "messages"): {
        "hosted": ["web_search", "code_execution"],
        "local": ["bash", "text_editor", "computer_use"],
    },
    ("deepseek", "chat_completions"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    ("deepseek", "messages"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    ("openrouter", "chat_completions"): {
        "hosted": ["web_search"],
        "local": ["function_calling"],
    },
    ("google", "generate_content"): {
        "hosted": ["web_search", "code_execution"],
        "local": ["function_calling"],
    },
    ("google", "chat_completions"): {
        "hosted": [],
        "local": ["function_calling"],
    },
}


def endpoint_for(provider: str, protocol: str, path: str) -> Endpoint:
    """Build an Endpoint for a provider surface, populated with its tool sets."""
    entry = TOOLS.get((provider, protocol), {})
    return Endpoint(
        protocol=protocol,
        endpoint=path,
        hosted_tools=list(entry.get("hosted", [])),
        local_tools=list(entry.get("local", [])),
    )

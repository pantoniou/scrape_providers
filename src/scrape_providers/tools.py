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

``protocol`` is normally a model wire format, but a provider may also publish a
standalone tool API — a raw JSON endpoint an agent or RAG pipeline calls itself,
with no model in the request. DeepSeek's Search and Fetch APIs are keyed that way
(``search``, ``fetch``), so they appear in the catalog's ``endpoints`` list beside
the model surfaces instead of being mistaken for a built-in chat tool.
"""

from __future__ import annotations

from .models import Endpoint, EndpointCapabilities

# (provider, protocol) -> {"hosted": [...], "local": [...]}
TOOLS: dict[tuple[str, str], dict[str, list[str]]] = {
    ("openai", "responses"): {
        "hosted": ["web_search", "file_search", "code_interpreter", "image_generation"],
        "local": ["local_shell", "computer_use", "function_calling", "mcp"],
    },
    # Chat Completions runs web search through `web_search_options`. Only the
    # search models accept it; the catalog records tools per endpoint, not
    # per model.
    ("openai", "chat_completions"): {
        "hosted": ["web_search"],
        "local": ["function_calling"],
    },
    ("anthropic", "messages"): {
        "hosted": ["web_search", "code_execution"],
        "local": ["bash", "text_editor", "computer_use"],
    },
    # Chat Completions has no server-side search: there is no
    # `web_search_options` equivalent, and the docs' feature table lists none.
    ("deepseek", "chat_completions"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    # The Responses API accepts one built-in tool, `web_search`: the provider
    # runs the search, adds the results to the context and continues the
    # response. It is `deepseek-v4-pro` only (`deepseek-flash` ignores it) — but
    # the catalog records tools per endpoint, not per model, as with OpenAI's
    # search models.
    ("deepseek", "responses"): {
        "hosted": ["web_search"],
        "local": ["function_calling"],
    },
    # The Anthropic-format surface accepts Anthropic's server-side web search:
    # both `server_tool_use` input items and `web_search_tool_result` content
    # blocks are supported.
    ("deepseek", "messages"): {
        "hosted": ["web_search"],
        "local": ["function_calling"],
    },
    # DeepSeek's own Search & Fetch APIs: plain JSON, billed per successful
    # request rather than per token, and usable with any model (or none). They
    # are what the harness integrations call server-side.
    ("deepseek", "search"): {
        "hosted": ["web_search"],
        "local": [],
    },
    ("deepseek", "fetch"): {
        "hosted": ["web_fetch"],
        "local": [],
    },
    ("fireworks", "chat_completions"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    ("fireworks", "responses"): {
        "hosted": [],
        "local": ["function_calling", "mcp"],
    },
    ("fireworks", "messages"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    # OpenCode Zen is a gateway: it forwards to each model's vendor, so the
    # surface offers the wire protocol's tool plumbing but no hosted tools of
    # its own.
    ("opencode", "chat_completions"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    ("opencode", "responses"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    ("opencode", "messages"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    ("opencode", "generate_content"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    # Muse Spark is always a reasoning model; web search is a hosted tool but
    # only reachable via the Responses API, not Chat Completions (see the docs'
    # `web_search_options` note).
    ("meta", "chat_completions"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    ("meta", "responses"): {
        "hosted": ["web_search"],
        "local": ["function_calling"],
    },
    ("meta", "messages"): {
        "hosted": [],
        "local": ["function_calling"],
    },
    ("openrouter", "chat_completions"): {
        "hosted": ["web_search"],
        "local": ["function_calling"],
    },
    ("openrouter", "responses"): {
        "hosted": ["web_search"],
        "local": ["function_calling", "mcp"],
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

# Behavioral properties of each provider/protocol surface. These are deliberately
# separate from model capabilities and tool inventories. Omitted fields are
# unknown or sufficiently model-dependent that the endpoint cannot promise them.
CAPABILITIES: dict[tuple[str, str], dict[str, bool]] = {
    ("openai", "responses"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "strict_json_schema_supported": True,
        "shell_tool_supported": True,
        "mcp_supported": True,
        "server_side_conversation_state_supported": True,
        "previous_response_id_supported": True,
        "response_lifecycle_supported": True,
        "system_prompt_supported": True,
        "developer_role_supported": True,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
        "response_compaction_supported": True,
    },
    ("openai", "chat_completions"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "strict_json_schema_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": True,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
        "response_compaction_supported": False,
    },
    ("openrouter", "responses"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": True,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": True,
        "explicit_prompt_caching_supported": True,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
        "response_compaction_supported": False,
    },
    ("openrouter", "chat_completions"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": True,
        "explicit_prompt_caching_supported": True,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
    },
    ("anthropic", "messages"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "strict_json_schema_supported": True,
        "shell_tool_supported": True,
        "mcp_supported": True,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": False,
        "explicit_prompt_caching_supported": True,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
    },
    ("deepseek", "chat_completions"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": False,
        "automatic_prompt_caching_supported": True,
    },
    # The Responses API is explicitly stateless: no stored responses, no
    # `previous_response_id`, no conversations. `parallel_tool_calls` and
    # `store` are ignored (always on / always false), `text.format` is fully
    # supported, `reasoning.effort` is honoured, `developer` messages are
    # accepted (treated as `user`), and caching is automatic — which is why
    # `prompt_cache_key` is not supported.
    ("deepseek", "responses"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "strict_json_schema_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": True,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
        "response_compaction_supported": False,
    },
    ("deepseek", "messages"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": False,
        "automatic_prompt_caching_supported": True,
    },
    ("fireworks", "chat_completions"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "strict_json_schema_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": False,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
    },
    ("fireworks", "responses"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "strict_json_schema_supported": True,
        "shell_tool_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": True,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
        "response_compaction_supported": False,
    },
    ("fireworks", "messages"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": False,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
    },
    # Muse Spark always reasons (`reasoning_effort: "none"` is HTTP 400) and
    # rejects `logprobs`, so `strict_json_schema_supported` (via `response_format`)
    # and `reasoning_controls_supported` are the model's fixed behavior, not a
    # per-call option.
    ("meta", "chat_completions"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "strict_json_schema_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
    },
    ("meta", "responses"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "strict_json_schema_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "previous_response_id_supported": True,
        "system_prompt_supported": True,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
    },
    ("meta", "messages"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": False,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
    },
    # Zen forwards to each model's vendor, so anything vendor-dependent
    # (caching mode, reasoning controls, server-side state) is left unstated.
    ("opencode", "chat_completions"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "system_prompt_supported": True,
    },
    ("opencode", "responses"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": True,
    },
    ("opencode", "messages"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": False,
    },
    ("opencode", "generate_content"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": False,
    },
    ("google", "generate_content"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": False,
        "explicit_prompt_caching_supported": True,
        "automatic_prompt_caching_supported": True,
        "reasoning_controls_supported": True,
    },
    ("google", "chat_completions"): {
        "streaming_supported": True,
        "function_calling_supported": True,
        "parallel_tool_calls_supported": True,
        "tool_choice_supported": True,
        "shell_tool_supported": False,
        "mcp_supported": False,
        "server_side_conversation_state_supported": False,
        "previous_response_id_supported": False,
        "response_lifecycle_supported": False,
        "system_prompt_supported": True,
        "developer_role_supported": False,
        "reasoning_controls_supported": True,
    },
}


def endpoint_for(provider: str, protocol: str, path: str) -> Endpoint:
    """Build an Endpoint with its curated tools and behavioral capabilities."""
    entry = TOOLS.get((provider, protocol), {})
    capabilities = CAPABILITIES.get((provider, protocol), {})
    return Endpoint(
        protocol=protocol,
        endpoint=path,
        hosted_tools=list(entry.get("hosted", [])),
        local_tools=list(entry.get("local", [])),
        capabilities=EndpointCapabilities(**capabilities),
    )

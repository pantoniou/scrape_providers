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

from .models import Endpoint, EndpointCapabilities

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

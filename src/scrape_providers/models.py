"""Normalized data model shared across all provider scrapers.

The YAML catalog is a serialization of these types, not of raw provider
responses. Provider-specific scrapers are responsible for mapping their data
into this model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Pricing(BaseModel):
    """Per-model pricing, normalized to USD per million tokens where applicable."""

    currency: str = "USD"
    unit: str = "per_million_tokens"
    input: float | None = None
    output: float | None = None
    # Provider-specific extras (e.g. cached input, image, audio) that don't fit
    # the common fields above.
    extra: dict[str, float] = Field(default_factory=dict)


class ArenaScore(BaseModel):
    """LMArena (Chatbot Arena) overall-leaderboard standing for a model."""

    rank: int
    elo: float
    votes: int | None = None


class Model(BaseModel):
    """A single model offered by a provider."""

    id: str
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    modalities: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    open_source: bool = False
    pricing: Pricing | None = None
    arena: ArenaScore | None = None
    # Canonical agent harness(es) that natively drive this model (gpt -> codex,
    # claude -> claude_code). Computed in emit from the model's vendor, not by
    # scrapers; see agent_profiles / emit.build_catalog.
    agents: list[str] = Field(default_factory=list)


class Agent(BaseModel):
    """A coding-agent harness, fully described for the catalog.

    Harness-level and model-agnostic: any function-calling model can be driven
    by it. The data is curated (`agent_profiles.py`) plus vendored captures
    (system prompt + tool schemas), not scraped from a provider.
    """

    name: str
    # The company/project that builds the agent (OpenAI, Anthropic, sst).
    developer: str | None = None
    # The catalog provider key the agent natively targets (openai, anthropic);
    # None for model-agnostic harnesses (opencode).
    native_provider: str | None = None
    protocol: str | None = None
    system_prompt: str | None = None
    # tool name -> {description?, schema}
    tools: dict[str, dict] = Field(default_factory=dict)


class EndpointCapabilities(BaseModel):
    """Behavioral capabilities of one provider API surface.

    These describe the provider/protocol contract, not an individual model.
    ``None`` means unknown or model-dependent and is pruned from catalog output;
    explicit ``False`` means the surface is known not to provide the behavior.
    """

    streaming_supported: bool | None = None
    function_calling_supported: bool | None = None
    parallel_tool_calls_supported: bool | None = None
    tool_choice_supported: bool | None = None
    strict_json_schema_supported: bool | None = None
    shell_tool_supported: bool | None = None
    mcp_supported: bool | None = None
    server_side_conversation_state_supported: bool | None = None
    previous_response_id_supported: bool | None = None
    response_lifecycle_supported: bool | None = None
    system_prompt_supported: bool | None = None
    developer_role_supported: bool | None = None
    explicit_prompt_caching_supported: bool | None = None
    automatic_prompt_caching_supported: bool | None = None
    reasoning_controls_supported: bool | None = None


class Endpoint(BaseModel):
    """One API surface a provider exposes: a wire protocol and its request path.

    A provider may offer the same models over several protocols (e.g. OpenAI via
    both ``chat_completions`` and ``responses``; DeepSeek via ``chat_completions``
    and an Anthropic-format ``messages`` endpoint).
    """

    protocol: str  # "chat_completions", "responses", or "messages"
    endpoint: str  # path relative to the provider root_url
    # Built-in tools this surface exposes, split by execution site (see tools.py):
    # hosted = run on the provider; local = the caller executes (bash, etc.).
    hosted_tools: list[str] = Field(default_factory=list)
    local_tools: list[str] = Field(default_factory=list)
    # Provider/protocol behavior, separate from intrinsic model capabilities.
    capabilities: EndpointCapabilities = Field(default_factory=EndpointCapabilities)


class Provider(BaseModel):
    """A provider and the catalog of models scraped from it."""

    name: str
    # API root URL; combined with an endpoint path to form the full request URL
    # (e.g. root "https://api.anthropic.com" + "/v1/messages").
    root_url: str | None = None
    # The protocol/endpoint surfaces this provider exposes for its models.
    endpoints: list[Endpoint] = Field(default_factory=list)
    models: list[Model] = Field(default_factory=list)

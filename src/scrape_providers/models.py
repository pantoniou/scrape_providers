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


class Provider(BaseModel):
    """A provider and the catalog of models scraped from it."""

    name: str
    # API root URL; combined with an endpoint path to form the full request URL
    # (e.g. root "https://api.anthropic.com" + "/v1/messages").
    root_url: str | None = None
    # The protocol/endpoint surfaces this provider exposes for its models.
    endpoints: list[Endpoint] = Field(default_factory=list)
    models: list[Model] = Field(default_factory=list)

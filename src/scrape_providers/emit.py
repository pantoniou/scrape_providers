"""Deterministic YAML emission of the scraped catalog.

Keys are emitted in a stable order and ``None``/empty values are dropped so that
run-to-run git diffs reflect real catalog changes (new models, price changes)
rather than serialization noise.
"""

from __future__ import annotations

from typing import Any

import yaml

from . import agent_profiles
from .canonical import canonical_id
from .models import Provider


def _prune(value: Any) -> Any:
    """Recursively drop None values and empty containers."""
    if isinstance(value, dict):
        out = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, {}, [])}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def build_catalog(providers: list[Provider], *, include_agents: bool = True) -> dict:
    """Split the scraped data into sections.

    ``models`` is a list of each model's intrinsic capabilities (with its
    canonical id as ``name``), sorted by name. ``providers`` lists which models
    each provider serves and references them by ``canonical_id``. ``endpoint`` and
    ``protocol`` are constant for a provider (each provider speaks one API), so
    they sit at the provider level; each model offering carries only what varies
    per model: ``canonical_id``, the provider's own ``provider_model_id``, and
    ``pricing``. A model served by several providers appears once under ``models``.

    When ``include_agents``, an ``agents`` section fully describes each known
    coding-agent harness (developer, native provider, system prompt, tool
    schemas), and each model is annotated with the canonical agent(s) that
    natively drive it (gpt -> codex, claude -> claude_code).
    """
    models: dict[str, dict] = {}
    provider_entries: list[dict] = []
    # native provider key -> agent name(s); used to tag models with their agent.
    prov_agents = agent_profiles.native_provider_agents() if include_agents else {}
    model_agents: dict[str, set[str]] = {}
    for provider in providers:
        offerings = []
        for m in provider.models:
            cid = canonical_id(m.id)
            # Intrinsic capabilities — shared across whoever serves the model.
            models.setdefault(
                cid,
                {
                    "display_name": m.display_name,
                    "context_window": m.context_window,
                    "max_output_tokens": m.max_output_tokens,
                    "modalities": m.modalities,
                    "capabilities": m.capabilities,
                    "open_source": m.open_source,
                    "arena": m.arena.model_dump() if m.arena else None,
                },
            )
            if include_agents:
                # A model is driven by agent A when its native provider serves it
                # (bare id, e.g. openai serves `gpt-5.5`) or its id carries that
                # vendor prefix (OpenRouter's `openai/gpt-5.5`).
                vendor = m.id.split("/")[0].lower() if "/" in m.id else None
                tags = set(prov_agents.get(provider.name, ()))
                if vendor:
                    tags |= set(prov_agents.get(vendor, ()))
                if tags:
                    model_agents.setdefault(cid, set()).update(tags)
            offerings.append(
                {
                    "canonical_id": cid,
                    "provider_model_id": m.id,
                    "pricing": m.pricing.model_dump() if m.pricing else None,
                }
            )
        provider_entries.append(
            {
                "name": provider.name,
                "root_url": provider.root_url,
                "endpoints": [e.model_dump() for e in provider.endpoints],
                "models": offerings,
            }
        )
    # Emit models as a list (canonical id as `name`), sorted by name for
    # deterministic output; field order is intentional (see sort_keys=False).
    model_list = [
        {"name": cid, **models[cid], "agents": sorted(model_agents.get(cid, ()))}
        for cid in sorted(models)
    ]
    catalog = {"models": model_list, "providers": provider_entries}
    if include_agents:
        catalog["agents"] = agent_profiles.build_agents()
    return catalog


def pruned_catalog(providers: list[Provider], *, include_agents: bool = True) -> dict:
    """The catalog dict exactly as serialized (empty/None values dropped)."""
    return _prune(build_catalog(providers, include_agents=include_agents))


def to_yaml(providers: list[Provider], *, include_agents: bool = True) -> str:
    catalog = pruned_catalog(providers, include_agents=include_agents)
    # sort_keys=False to keep the intentional field order (name, root_url,
    # endpoint, protocol, models); determinism comes from the sorted models map.
    return yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _tokens(n: int | None) -> str:
    if n is None:
        return "—"
    for divisor, suffix in ((1_000_000, "M"), (1000, "K")):
        if n >= divisor:
            value = n / divisor
            # Whole numbers render without a decimal; otherwise up to 2 places.
            text = f"{value:.0f}" if value == int(value) else f"{value:.2f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return str(n)


def _money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:g}"


def _agent_tags(providers: list[Provider]) -> dict[str, list[str]]:
    """Canonical-id -> sorted agent names, mirroring build_catalog's tagging."""
    prov_agents = agent_profiles.native_provider_agents()
    tags: dict[str, set[str]] = {}
    for provider in providers:
        for m in provider.models:
            vendor = m.id.split("/")[0].lower() if "/" in m.id else None
            hit = set(prov_agents.get(provider.name, ()))
            if vendor:
                hit |= set(prov_agents.get(vendor, ()))
            if hit:
                tags.setdefault(canonical_id(m.id), set()).update(hit)
    return {cid: sorted(names) for cid, names in tags.items()}


def to_markdown(providers: list[Provider], *, include_agents: bool = True) -> str:
    """Render the catalog as human-readable Markdown.

    The Arena (Elo / rank) columns are only shown when at least one model in the
    catalog carries arena data, so non-``--arena`` output stays narrow. When
    ``include_agents``, a per-model Agents column (when any model has one) and a
    trailing Agents section summarize the agent harnesses.
    """
    show_arena = any(m.arena for p in providers for m in p.models)
    agent_tags = _agent_tags(providers) if include_agents else {}
    show_agents = bool(agent_tags)
    lines: list[str] = ["# AI Provider Catalog", ""]
    for provider in providers:
        lines.append(f"## {provider.name}")
        meta = []
        if provider.root_url:
            meta.append(f"`root_url: {provider.root_url}`")
        for e in provider.endpoints:
            chunk = f"{e.protocol} `{e.endpoint}`"
            if e.hosted_tools:
                chunk += f" hosted[{', '.join(e.hosted_tools)}]"
            if e.local_tools:
                chunk += f" local[{', '.join(e.local_tools)}]"
            meta.append(chunk)
        meta.append(f"{len(provider.models)} models")
        lines.append(" · ".join(meta))
        lines.append("")

        arena_head = " Arena # | Elo |" if show_arena else ""
        arena_sep = "--:|--:|" if show_arena else ""
        agent_head = " Agents |" if show_agents else ""
        agent_sep = "---|" if show_agents else ""
        lines.append(
            f"| Model | ID | OSS |{arena_head} Context | Max output | Input | Cached | Output | Modalities |{agent_head}"
        )
        lines.append(f"|---|---|:-:|{arena_sep}--:|--:|--:|--:|--:|---|{agent_sep}")
        for m in provider.models:
            price = m.pricing
            arena_cells = ""
            if show_arena:
                rank = f"#{m.arena.rank}" if m.arena else "—"
                elo = f"{m.arena.elo:.0f}" if m.arena else "—"
                arena_cells = f" {rank} | {elo} |"
            agent_cells = ""
            if show_agents:
                names = agent_tags.get(canonical_id(m.id))
                agent_cells = f" {', '.join(names) if names else '—'} |"
            cached = price.extra.get("cache_read") if price else None
            lines.append(
                "| {name} | `{id}` | {oss} |{arena} {ctx} | {out} | {pin} | {cache} | {pout} | {mods} |{agents}".format(
                    name=m.display_name or m.id,
                    id=m.id,
                    oss="✓" if m.open_source else "",
                    arena=arena_cells,
                    ctx=_tokens(m.context_window),
                    out=_tokens(m.max_output_tokens),
                    pin=_money(price.input) if price else "—",
                    cache=_money(cached),
                    pout=_money(price.output) if price else "—",
                    mods=", ".join(m.modalities) or "—",
                    agents=agent_cells,
                )
            )
        lines.append("")
        lines.append("*Prices in USD per million tokens.*")
        lines.append("")

    if include_agents:
        lines.extend(_markdown_agents())
    return "\n".join(lines).rstrip() + "\n"


def _markdown_agents() -> list[str]:
    """Render the Agents section: one block per harness, tool names not schemas."""
    lines = ["## Agents", ""]
    for agent in agent_profiles.build_agents():
        lines.append(f"### {agent['name']}")
        meta = []
        if agent.get("developer"):
            meta.append(f"developer: {agent['developer']}")
        if agent.get("native_provider"):
            meta.append(f"native provider: `{agent['native_provider']}`")
        else:
            meta.append("native provider: — (model-agnostic)")
        if agent.get("protocol"):
            meta.append(f"protocol: {agent['protocol']}")
        prompt = agent.get("system_prompt")
        meta.append(f"system prompt: {len(prompt)} chars" if prompt else "system prompt: —")
        lines.append(" · ".join(meta))
        lines.append("")
        tools = agent.get("tools") or {}
        if tools:
            lines.append(f"{len(tools)} tools: {', '.join(sorted(tools))}")
        else:
            lines.append("tools: — (no vendored capture)")
        lines.append("")
    return lines

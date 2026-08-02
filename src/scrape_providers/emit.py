"""Deterministic YAML emission of the scraped catalog.

Keys are emitted in a stable order and ``None``/empty values are dropped so that
run-to-run git diffs reflect real catalog changes (new models, price changes)
rather than serialization noise.
"""

from __future__ import annotations

import re
import sys
from typing import Any

import yaml

from . import agent_profiles
from .canonical import canonical_id
from .capabilities import normalize_capabilities, normalize_modalities
from .models import Model, Provider


def _prune(value: Any) -> Any:
    """Recursively drop None values and empty containers."""
    if isinstance(value, dict):
        out = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, {}, [])}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def _keys_of(node: Any) -> list[str]:
    """The addressable children of a node, for error messages."""
    if isinstance(node, dict):
        return [str(k) for k in node]
    if isinstance(node, list):
        return [str(item["name"]) for item in node if isinstance(item, dict) and "name" in item]
    return []


def _describe(names: list[str], limit: int = 12) -> str:
    if not names:
        return "nothing addressable"
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, … ({len(names)} total)"


def select_path(catalog: Any, path: str) -> Any:
    """Return the part of ``catalog`` addressed by a slash-separated ``path``.

    Mappings are indexed by key. Lists are indexed by the ``name`` of their
    entries, not by position — the catalog's lists are named collections
    (``agents/codex``, ``providers/openrouter``), and their order is an emission
    detail that must not become part of the address. Empty segments are ignored,
    so ``models`` and ``models/`` are the same path and ``""`` is the whole
    catalog.

    A model's canonical id may itself contain a slash on no current provider,
    but a provider model id can (``openai/gpt-5.5``), so paths address models by
    canonical id only. Raises ``KeyError`` naming the alternatives when a
    segment doesn't resolve.
    """
    node = catalog
    walked: list[str] = []
    for segment in [s for s in path.split("/") if s]:
        if isinstance(node, dict) and segment in node:
            node = node[segment]
        elif isinstance(node, list) and (
            hit := next(
                (i for i in node if isinstance(i, dict) and str(i.get("name")) == segment), None
            )
        ):
            node = hit
        else:
            where = "/".join(walked) or "<root>"
            raise KeyError(f"no {segment!r} under {where}; have: {_describe(_keys_of(node))}")
        walked.append(segment)
    return node


def _union(first: list[str], second: list[str]) -> list[str]:
    """Both lists, first-seen order preserved (deterministic given provider order)."""
    return list(dict.fromkeys([*first, *second]))


# How far a provider is from the model's vendor, for fields that can only take
# one value (a name, not a maximum). First-party providers serve only models they
# built, so if one of them serves the model it *is* the vendor and ranks 0; the
# resellers below rank behind it, by how close their naming is to the vendor's own:
# Fireworks republishes the model card's name, Zen uses a plain name too, while
# OpenRouter prefixes a vendor label ("MoonshotAI: Kimi K3") that the vendor
# doesn't use, so it ranks last.
_RESELLER_RANK = {"fireworks": 1, "opencode": 2, "openrouter": 3}
# Fields with no "strongest claim" to take, so the nearest provider wins instead.
_VENDOR_FIELDS = ("display_name", "arena")
# Parameter-count monikers, the only thing telling some models apart (gpt-oss-20b
# vs gpt-oss-120b, gemma-4-26b-a4b vs gemma-4-31b). Canonical ids always keep
# them, but a provider's display name may not.
_SIZE = re.compile(r"\d+(?:\.\d+)?\s*[bm]\b", re.IGNORECASE)


def _sizes(text: str) -> set[str]:
    return {match.lower().replace(" ", "") for match in _SIZE.findall(text)}


def _naming_rank(provider: str, name: str | None, cid: str) -> tuple[int, int]:
    """Preference for one provider's name of a model — lower is better.

    A name that keeps every size moniker in the canonical id sorts ahead of one
    that drops it, whoever published it: for a family that differs only by size,
    a name without the size doesn't identify the model at all. Distance from the
    vendor breaks the tie.
    """
    drops_size = bool(name is not None and not _sizes(cid) <= _sizes(name))
    return (int(drops_size), _RESELLER_RANK.get(provider, 0))


def _merge_capabilities(cid: str, views: list[tuple[str, Model]], unmapped: set[str]) -> dict:
    """Build one model's intrinsic entry from every provider that serves it.

    A first-party provider serves only models it built, so when one of them
    serves this model its description **is** the model: a reseller cannot route
    a capability the vendor doesn't have, and where the two disagree the vendor
    is right. The vendor's views alone are therefore authoritative for context
    window, max output, modalities and capabilities, and the resellers' claims
    are dropped here — they remain visible per route as the offerings'
    ``reported_*`` fields.

    With no first-party provider (open-weight models nobody in the catalog
    built) every reseller's view counts, and since each publishes a partial
    positive set the strongest claim wins: widest context window, union of
    modalities and capabilities.

    ``open_source`` is always an OR across all views, first-party included: a
    provider that doesn't mention weights leaves the field at its ``False``
    default, which is an absence of a claim rather than a claim of closedness.
    ``display_name`` and ``arena`` are single values with no "strongest"
    answer, so ``_naming_rank`` picks one instead.

    Modalities and capabilities are then mapped onto the controlled vocabulary
    in :mod:`capabilities`, since providers describe the same abilities in
    mutually incomparable words. Terms it recognizes neither as a capability nor
    as a deliberate non-capability are collected in ``unmapped`` for the caller
    to report.
    """
    authoritative = [v for v in views if v[0] not in _RESELLER_RANK] or views
    models = [m for _, m in authoritative]
    contexts = [m.context_window for m in models if m.context_window is not None]
    outputs = [m.max_output_tokens for m in models if m.max_output_tokens is not None]
    modalities: list[str] = []
    capabilities: list[str] = []
    for m in models:
        modalities = _union(modalities, m.modalities)
        capabilities = _union(capabilities, m.capabilities)
    # Only the merged entry speaks the canonical vocabulary; each offering's
    # reported_* keeps its provider's own words (see capabilities.py).
    modalities, unknown_mods = normalize_modalities(modalities)
    capabilities, unknown_caps = normalize_capabilities(capabilities)
    unmapped.update(unknown_mods)
    unmapped.update(unknown_caps)
    named = sorted(views, key=lambda v: _naming_rank(v[0], v[1].display_name, cid))
    return {
        "display_name": next((m.display_name for _, m in named if m.display_name), None),
        "context_window": max(contexts) if contexts else None,
        "max_output_tokens": max(outputs) if outputs else None,
        "modalities": modalities,
        "capabilities": capabilities,
        "open_source": any(m.open_source for _, m in views),
        "arena": next((m.arena.model_dump() for _, m in named if m.arena), None),
    }


def build_catalog(providers: list[Provider], *, include_agents: bool = True) -> dict:
    """Split the scraped data into sections.

    ``models`` is a list of each model's intrinsic capabilities (with its
    canonical id as ``name``), sorted by name. ``providers`` lists which models
    each provider serves and references them by ``canonical_id``. ``endpoint`` and
    ``protocol`` are constant for a provider (each provider speaks one API), so
    they sit at the provider level; each model offering carries only what varies
    per model: ``canonical_id``, the provider's own ``provider_model_id``, and
    ``pricing`` plus the ``reported_*`` fields. A model served by several
    providers appears once under ``models``, described by its vendor where the
    catalog has one (see :func:`_merge_capabilities`).

    When ``include_agents``, an ``agents`` section fully describes each known
    coding-agent harness (developer, native provider, system prompt, tool
    schemas), and each model is annotated with the canonical agent(s) that
    natively drive it (gpt -> codex, claude -> claude_code).
    """
    # canonical id -> every (provider name, model) that describes it, merged into
    # the intrinsic entry once all providers have been walked.
    views: dict[str, list[tuple[str, Model]]] = {}
    provider_entries: list[dict] = []
    # native provider key -> agent name(s); used to tag models with their agent.
    prov_agents = agent_profiles.native_provider_agents() if include_agents else {}
    model_agents: dict[str, set[str]] = {}
    for provider in providers:
        offerings = []
        for m in provider.models:
            cid = canonical_id(m.id)
            views.setdefault(cid, []).append((provider.name, m))
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
                    # What *this* provider publishes for the model, as opposed to
                    # the vendor's authoritative entry under `models`. Reported,
                    # not supported: providers
                    # only ever publish a positive set, in their own vocabulary
                    # (OpenRouter lists request parameters, Anthropic lists
                    # features, Zen publishes nothing), so a capability missing
                    # here is not evidence the route lacks it.
                    "reported_modalities": list(m.modalities),
                    "reported_capabilities": list(m.capabilities),
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
    unmapped: set[str] = set()
    model_list = [
        {
            "name": cid,
            **_merge_capabilities(cid, views[cid], unmapped),
            "agents": sorted(model_agents.get(cid, ())),
        }
        for cid in sorted(views)
    ]
    if unmapped:
        # Dropped, but never silently: an unrecognized term is a provider's new
        # vocabulary waiting to be classified in capabilities.py.
        print(
            f"warning: dropped {len(unmapped)} unrecognized capability/modality "
            f"term(s): {', '.join(sorted(unmapped))}",
            file=sys.stderr,
        )
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

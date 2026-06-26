"""Command-line entry point: scrape providers and emit a YAML catalog."""

from __future__ import annotations

import argparse
import json
import sys

import httpx

import yaml

from . import agent_profiles
from . import arena as arena_mod
from . import registry
from . import schema as schema_mod
from . import curation
from .curation import curate
from .emit import pruned_catalog, to_markdown, to_yaml
from .models import Provider


def _validate_file(path: str) -> int:
    """Validate a catalog file (YAML or JSON) against the schema. Returns exit code."""
    import jsonschema

    try:
        with open(path, encoding="utf-8") as fh:
            catalog = yaml.safe_load(fh)  # also parses JSON
    except (OSError, yaml.YAMLError) as exc:
        print(f"could not read {path}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(catalog, dict):
        print(f"{path}: expected a mapping at the top level", file=sys.stderr)
        return 1

    try:
        schema_mod.validate_catalog(catalog)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        print(f"{path}: invalid at {location}: {exc.message}", file=sys.stderr)
        return 1

    print(f"{path}: valid")
    return 0


def _set_curated(path: str) -> int:
    """Load a curated mapping from a file, validate it, and persist it."""
    try:
        with open(path, encoding="utf-8") as fh:
            mapping = yaml.safe_load(fh)  # also parses JSON
    except (OSError, yaml.YAMLError) as exc:
        print(f"could not read {path}: {exc}", file=sys.stderr)
        return 1
    try:
        saved = curation.save_curated(mapping)
    except ValueError as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return 1
    print(f"curated list saved to {saved}")
    return 0


def _agent_tool_schema(arg: str) -> int:
    """Print vendored tool schema(s). arg is AGENT or AGENT/TOOL (split on first '/')."""
    agent, _, tool = arg.partition("/")
    try:
        agent_profiles.get(agent)  # validate the agent name
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    details = agent_profiles.tool_details(agent)
    if not details:
        print(
            f"no vendored schemas for {agent!r}; add agent_schemas/{agent}.json "
            "(see agent_schemas/README.md to capture)",
            file=sys.stderr,
        )
        return 1
    if tool:
        if tool not in details:
            print(
                f"{agent!r} has no vendored schema for tool {tool!r}; "
                f"have: {', '.join(sorted(details))}",
                file=sys.stderr,
            )
            return 1
        print(json.dumps(details[tool], indent=2))
    else:
        print(json.dumps(details, indent=2))
    return 0


def _tool_summary(desc: str) -> str:
    """First line of a tool description, truncated for a one-line listing."""
    line = (desc or "").strip().splitlines()[0] if desc.strip() else ""
    return line if len(line) <= 100 else line[:99].rstrip() + "…"


def _print_tool_lines(agent: str, indent: str = "") -> None:
    """Print an agent's tools as an aligned `name  description` table."""
    names = agent_profiles.tool_names(agent)
    descs = agent_profiles.tool_descriptions(agent)
    width = max((len(n) for n in names), default=0)
    for tool in names:
        summary = _tool_summary(descs.get(tool, ""))
        if summary:
            print(f"{indent}{tool:<{width}}  {summary}")
        else:
            print(f"{indent}{tool}")


def _list_agent_tools(agent: str) -> int:
    """List a harness's standard tools, or all harnesses when agent == '__all__'."""
    if agent == "__all__":
        for name in agent_profiles.available():
            p = agent_profiles.get(name)
            src = "captured" if agent_profiles.has_capture(name) else "curated"
            print(
                f"# {name} — {p['description']} (protocol: {p['protocol']}) "
                f"[{src}] — {p['source']}"
            )
            _print_tool_lines(name, indent="  ")
        return 0
    try:
        agent_profiles.get(agent)  # validate the name
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    _print_tool_lines(agent)
    return 0


def _agent_system_prompt(agent: str) -> int:
    """Print an agent's vendored system prompt."""
    try:
        agent_profiles.get(agent)  # validate the agent name
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    prompt = agent_profiles.system_prompt(agent)
    if prompt is None:
        print(
            f"no vendored system prompt for {agent!r}; capture one with "
            f"scripts/capture/capture-{agent.replace('_', '-')}.sh "
            "(see agent_schemas/README.md)",
            file=sys.stderr,
        )
        return 1
    sys.stdout.write(prompt if prompt.endswith("\n") else prompt + "\n")
    return 0


def _list_provider_models(name: str, *, curated: bool) -> int:
    """Scrape a single provider and print its model ids, one per line."""
    try:
        scraper_cls = registry.get(name)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)  # unwrap KeyError's quote-wrapped str
        return 1
    with scraper_cls() as scraper:
        provider = scraper.scrape()
    if curated:
        provider = curate(provider)
    for model in provider.models:
        print(model.id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scrape-providers",
        description="Scrape AI providers for endpoints, characteristics, and pricing.",
    )
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        metavar="NAME",
        help="Provider to scrape (repeatable). Defaults to all registered providers.",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("yaml", "markdown"),
        default="yaml",
        help="Output format (default: yaml).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write the catalog to this file instead of stdout.",
    )
    default_list = "; ".join(
        f"{prov}: {', '.join(ids)}" for prov, ids in curation.DEFAULT_CURATED.items()
    )
    parser.add_argument(
        "--curated",
        action="store_true",
        help=(
            "Keep only the curated latest/flagship models. Defaults (override with "
            f"--set-curated) — {default_list}"
        ),
    )
    parser.add_argument(
        "--arena",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate models with LMArena Elo / rank (default: on; --no-arena to skip "
        "the extra network fetch).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the catalog against the JSON schema before emitting.",
    )
    parser.add_argument(
        "--validate-file",
        metavar="FILE",
        help="Validate an existing catalog file (YAML or JSON) against the schema and exit.",
    )
    parser.add_argument(
        "--print-curated",
        action="store_true",
        help="Print the active curated model list (YAML) and exit.",
    )
    parser.add_argument(
        "--set-curated",
        metavar="FILE",
        help="Persist the curated model list from a YAML/JSON mapping file and exit.",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Print the catalog JSON schema and exit.",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and exit.",
    )
    parser.add_argument(
        "--list-provider-models",
        metavar="PROVIDER",
        help="List the model ids a single provider serves and exit (honors --curated).",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available agent harnesses and exit.",
    )
    parser.add_argument(
        "--list-agent-tools",
        nargs="?",
        const="__all__",
        metavar="AGENT",
        help="List the standard function-calling tools an agent harness exposes "
        "(codex, claude_code). With no AGENT, list all harnesses. Then exit.",
    )
    parser.add_argument(
        "--agent-tool-schema",
        metavar="AGENT[/TOOL]",
        help="Print vendored JSON schemas for an agent's tools (all, or one TOOL). "
        "See agent_schemas/README.md for how to capture them.",
    )
    parser.add_argument(
        "--agent-system-prompt",
        metavar="AGENT",
        help="Print an agent harness's vendored system prompt (captured alongside "
        "its tools). See agent_schemas/README.md for how to capture.",
    )
    parser.add_argument(
        "--show",
        metavar="PROVIDER/MODEL",
        help="Output a single model in the chosen format. Split on the first '/', "
        "so the model id may itself contain slashes (e.g. openrouter/z-ai/glm-5.2).",
    )
    args = parser.parse_args(argv)

    if args.schema:
        print(json.dumps(schema_mod.load_schema(), indent=2))
        return 0

    if args.print_curated:
        sys.stdout.write(yaml.safe_dump(curation.load_curated(), sort_keys=True))
        return 0

    if args.set_curated:
        return _set_curated(args.set_curated)

    if args.validate_file:
        return _validate_file(args.validate_file)

    if args.list_providers:
        print("\n".join(registry.available()))
        return 0

    if args.list_provider_models:
        return _list_provider_models(args.list_provider_models, curated=args.curated)

    if args.list_agents:
        print("\n".join(agent_profiles.available()))
        return 0

    if args.list_agent_tools is not None:
        return _list_agent_tools(args.list_agent_tools)

    if args.agent_tool_schema:
        return _agent_tool_schema(args.agent_tool_schema)

    if args.agent_system_prompt:
        return _agent_system_prompt(args.agent_system_prompt)

    show_model: str | None = None
    if args.show:
        if "/" not in args.show:
            print("--show expects PROVIDER/MODEL (e.g. deepseek/deepseek-v4-pro)", file=sys.stderr)
            return 1
        # Provider names never contain '/', so split on the first one; the
        # remainder is the model id, which may itself contain slashes.
        show_provider, show_model = args.show.split("/", 1)
        names = [show_provider]
    else:
        names = args.providers or registry.available()

    results: list[Provider] = []
    for name in names:
        try:
            scraper_cls = registry.get(name)
        except KeyError as exc:
            print(exc.args[0], file=sys.stderr)
            return 1
        with scraper_cls() as scraper:
            provider = scraper.scrape()
        if args.curated:
            provider = curate(provider)
        if show_model is not None:
            kept = [m for m in provider.models if m.id == show_model]
            if not kept:
                print(
                    f"provider {name!r} has no model {show_model!r}", file=sys.stderr
                )
                return 1
            provider = provider.model_copy(update={"models": kept})
        results.append(provider)

    if args.arena:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            arena_mod.annotate(results, arena_mod.fetch_scores(client))

    if args.validate:
        import jsonschema

        try:
            schema_mod.validate_catalog(pruned_catalog(results))
        except jsonschema.ValidationError as exc:
            print(f"catalog failed schema validation: {exc.message}", file=sys.stderr)
            return 1

    output = to_markdown(results) if args.format == "markdown" else to_yaml(results)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

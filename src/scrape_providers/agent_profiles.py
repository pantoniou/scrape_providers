"""Curated tool/function profiles for well-known agent harnesses.

These are the standard tools popular coding agents expose to a model via function
calling. They are harness-level and model-agnostic — any function-calling-capable
model can be driven by them — so they live here rather than under a provider.

* Codex tool specs are open source (``codex-rs/core/src/tools/handlers/*_spec.rs``
  in the repo below); the core defaults are listed here.
* Claude Code is not open source; its tool names are documented / introspectable.

Names are accurate; this module does not (yet) carry the full JSON parameter
schemas. Update as harnesses change.
"""

from __future__ import annotations

import json
from importlib.resources import files

AGENT_PROFILES: dict[str, dict] = {
    "codex": {
        "description": "OpenAI Codex CLI",
        "developer": "OpenAI",
        "provider": "openai",  # native catalog provider it targets
        "protocol": "responses",
        "source": "https://github.com/openai/codex",
        "tools": [
            "shell",  # run a shell command
            "apply_patch",  # create/update/delete files via a structured diff
            "update_plan",  # maintain a step-by-step TODO plan
        ],
    },
    "opencode": {
        "description": "opencode (sst) — open-source terminal coding agent",
        "developer": "sst",
        "provider": None,  # model-agnostic: no single native provider
        "protocol": "various",  # provider-agnostic, via the AI SDK
        "source": "https://github.com/sst/opencode",
        "tools": [
            "bash",
            "read",
            "write",
            "edit",
            "glob",
            "grep",
            "webfetch",
            "websearch",
            "todowrite",
            "task",
            "lsp",
            "skill",
            "apply_patch",
            "plan_exit",
            "question",
        ],
    },
    "claude_code": {
        "description": "Anthropic Claude Code",
        "developer": "Anthropic",
        "provider": "anthropic",  # native catalog provider it targets
        "protocol": "messages",
        "source": "https://docs.claude.com/en/docs/claude-code",
        "tools": [
            "Bash",
            "Read",
            "Edit",
            "Write",
            "Glob",
            "Grep",
            "Task",
            "TodoWrite",
            "WebFetch",
            "WebSearch",
            "NotebookEdit",
            "BashOutput",
            "KillShell",
            "AskUserQuestion",
            "Skill",
        ],
    },
}


def available() -> list[str]:
    return sorted(AGENT_PROFILES)


def get(name: str) -> dict:
    try:
        return AGENT_PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown agent {name!r}; available: {', '.join(available())}") from None


def agent_catalog_entry(name: str) -> dict:
    """Assemble the full catalog description of one agent from vendored data."""
    profile = get(name)
    return {
        "name": name,
        "developer": profile.get("developer"),
        "native_provider": profile.get("provider"),
        "protocol": profile.get("protocol"),
        "system_prompt": system_prompt(name),
        "tools": tool_details(name),
    }


def build_agents() -> list[dict]:
    """All known agents as catalog entries (sorted by name)."""
    return [agent_catalog_entry(name) for name in available()]


def native_provider_agents() -> dict[str, list[str]]:
    """Map a native catalog provider key -> the agent name(s) that target it."""
    out: dict[str, list[str]] = {}
    for name in available():
        provider = get(name).get("provider")
        if provider:
            out.setdefault(provider, []).append(name)
    return out


def tool_details(agent: str) -> dict[str, dict]:
    """Return ``tool name -> {description?, schema}`` from a vendored capture."""
    descriptions = tool_descriptions(agent)
    out: dict[str, dict] = {}
    for name, schema in load_schemas(agent).items():
        entry: dict = {}
        desc = descriptions.get(name)
        if desc:
            entry["description"] = desc
        entry["schema"] = schema
        out[name] = entry
    return out


def tool_names(agent: str) -> list[str]:
    """Tool names for an agent.

    Derived from the vendored capture if one exists (ground truth), otherwise the
    curated fallback list in ``AGENT_PROFILES``.
    """
    captured = load_schemas(agent)
    if captured:
        return sorted(captured)
    return list(AGENT_PROFILES.get(agent, {}).get("tools", []))


def has_capture(agent: str) -> bool:
    return bool(load_schemas(agent))


def system_prompt(agent: str) -> str | None:
    """Return the vendored system prompt for an agent, or None if not captured.

    Prompts live in ``agent_prompts/<agent>.txt`` (captured alongside the tools;
    see ``agent_schemas/README.md``).
    """
    path = files("scrape_providers").joinpath("agent_prompts", f"{agent}.txt")
    if not path.is_file():
        return None
    return path.read_text("utf-8")


def has_system_prompt(agent: str) -> bool:
    return system_prompt(agent) is not None


def _load_raw(agent: str) -> object | None:
    """Load the raw vendored capture for an agent, or None if there's no file."""
    path = files("scrape_providers").joinpath("agent_schemas", f"{agent}.json")
    if not path.is_file():
        return None
    return json.loads(path.read_text("utf-8"))


def load_schemas(agent: str) -> dict[str, dict]:
    """Return ``tool name -> JSON schema`` from a vendored capture, or {} if none.

    Captures live in ``agent_schemas/<agent>.json`` (see that dir's README). The
    file may be the raw ``tools`` array a harness sends, a ``{"tools": [...]}``
    wrapper, or an already-indexed ``{name: schema}`` map; :func:`index_tools`
    normalizes all three.
    """
    raw = _load_raw(agent)
    return index_tools(raw) if raw is not None else {}


def tool_descriptions(agent: str) -> dict[str, str]:
    """Return ``tool name -> description`` from a vendored capture, or {} if none."""
    raw = _load_raw(agent)
    if isinstance(raw, dict) and "tools" in raw and isinstance(raw["tools"], list):
        raw = raw["tools"]
    if not isinstance(raw, list):
        return {}
    out: dict[str, str] = {}
    for tool in raw:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = tool.get("name") or fn.get("name") or tool.get("type")
        if name:
            out[name] = tool.get("description") or fn.get("description") or ""
    return out


def index_tools(data: object) -> dict[str, dict]:
    """Normalize a captured tools payload into ``{tool name: parameter schema}``."""
    if isinstance(data, dict) and "tools" in data and isinstance(data["tools"], (list, dict)):
        data = data["tools"]
    if isinstance(data, list):
        out: dict[str, dict] = {}
        for tool in data:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
            # Built-in/hosted tools (web_search, image_generation, ...) are typed
            # entries with no name; key those by their `type` so they aren't lost.
            name = tool.get("name") or fn.get("name") or tool.get("type")
            if not name:
                continue
            schema = (
                tool.get("input_schema")  # Anthropic messages
                or tool.get("parameters")  # OpenAI chat/responses
                or fn.get("parameters")  # OpenAI function-wrapped
            )
            if schema is None:
                # Typed built-in: keep its config (everything but the name).
                schema = {k: v for k, v in tool.items() if k != "name"} or tool
            out[name] = schema
        return out
    if isinstance(data, dict):
        return data  # already {name: schema}
    return {}

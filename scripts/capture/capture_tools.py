"""mitmproxy addon: capture the `tools` array an agent sends to the model API.

Writes the tools payload (as the raw JSON array) to ``$CAPTURE_OUT`` whenever a
request to a known model endpoint carries one. If ``$CAPTURE_PROMPT_OUT`` is set,
the request's system prompt is written there too (as plain text). Run via the
launcher scripts in this directory, or directly:

    CAPTURE_OUT=out.json mitmdump -s capture_tools.py --listen-port 8080
"""

from __future__ import annotations

import json
import os

try:  # mitmproxy is only needed when run as an addon, not to import the helpers
    from mitmproxy import ctx, http
except ImportError:  # pragma: no cover
    ctx = http = None

OUT = os.environ.get("CAPTURE_OUT", "tools.json")
PROMPT_OUT = os.environ.get("CAPTURE_PROMPT_OUT")
DEBUG = bool(os.environ.get("CAPTURE_DEBUG"))


def _text_blocks(value) -> str:
    """Flatten a string or a list of {type:text, text:...} content blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def extract_system_prompt(body: dict) -> str:
    """Pull the system/developer instructions out of an LLM request body.

    Handles Anthropic Messages (``system``), OpenAI Responses (``instructions``
    plus ``input`` developer/system turns) and Chat Completions (``messages``).
    """
    parts = []
    # Anthropic Messages
    if "system" in body:
        parts.append(_text_blocks(body["system"]))
    # OpenAI Responses
    if isinstance(body.get("instructions"), str):
        parts.append(body["instructions"])
    # Leading system/developer turns in `messages` (chat) or `input` (responses)
    for key in ("messages", "input"):
        turns = body.get(key)
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            if turn.get("role") not in ("system", "developer"):
                break  # stop at the first non-system turn
            parts.append(_text_blocks(turn.get("content")))
    return "\n\n".join(p for p in parts if p).strip()


# Keys that indicate an LLM request body (so we don't grab some unrelated POST
# that happens to have a "tools" field). Host-agnostic on purpose: agents that
# use login-based auth talk to their own hosts (chatgpt.com, etc.), not the
# public api.* endpoints.
LLM_BODY_KEYS = ("messages", "input", "model", "system", "prompt", "contents")


def request(flow: http.HTTPFlow) -> None:
    host = flow.request.pretty_host
    if flow.request.method != "POST":
        if DEBUG:
            ctx.log.alert(f"[seen] {flow.request.method} {host}{flow.request.path}")
        return
    try:
        body = json.loads(flow.request.get_text() or "{}")
    except (ValueError, TypeError):
        return
    if not isinstance(body, dict):
        return
    if DEBUG:
        ctx.log.alert(f"[POST] {host}{flow.request.path}  keys={sorted(body)[:8]}")

    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return
    if not any(k in body for k in LLM_BODY_KEYS):
        return
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(tools, fh, indent=2)
    ctx.log.alert(f"captured {len(tools)} tools from {host}{flow.request.path} -> {OUT}")

    if PROMPT_OUT:
        prompt = extract_system_prompt(body)
        if prompt:
            with open(PROMPT_OUT, "w", encoding="utf-8") as fh:
                fh.write(prompt + "\n")
            ctx.log.alert(f"captured system prompt ({len(prompt)} chars) -> {PROMPT_OUT}")

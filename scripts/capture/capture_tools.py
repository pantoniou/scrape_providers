"""mitmproxy addon: capture the `tools` array an agent sends to the model API.

Writes the tools payload (as the raw JSON array) to ``$CAPTURE_OUT`` whenever a
request to a known model endpoint carries one. Run via the launcher scripts in
this directory, or directly:

    CAPTURE_OUT=out.json mitmdump -s capture_tools.py --listen-port 8080
"""

import json
import os

from mitmproxy import ctx, http

OUT = os.environ.get("CAPTURE_OUT", "tools.json")
DEBUG = bool(os.environ.get("CAPTURE_DEBUG"))

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

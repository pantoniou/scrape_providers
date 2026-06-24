# Vendored agent tool schemas

Drop a captured tools payload here as `<agent>.json` (e.g. `claude_code.json`,
`codex.json`, `opencode.json`). The agent name must match one in
`agent_profiles.py`.

`--agent-tool-schema <agent>[/<tool>]` reads these. No file → no schemas (names
still come from `--list-agent-tools`).

## How to capture

The full JSON schema for every tool is in the `tools` array of each request the
agent sends to the model API. Capture one with a proxy, then save the array here.

- **ccglass** (built for Claude Code / Codex): https://github.com/jianshuo/ccglass
- **mitmproxy** reverse proxy:
  ```bash
  mitmweb --mode reverse:https://api.anthropic.com --listen-port 8000
  # point the agent at it (e.g. HTTPS_PROXY + NODE_EXTRA_CA_CERTS for Claude Code),
  # run one trivial turn, then copy the request's `tools` array.
  ```

**Redact API keys / secrets** from anything you capture before saving.

## Accepted file shapes (any of these)

- the raw `tools` array: `[{"name": "...", "input_schema": {...}}, ...]`
  (Anthropic) or `[{"name": "...", "parameters": {...}}, ...]` (OpenAI)
- a wrapper: `{"tools": [ ... ]}`
- already indexed: `{"<tool>": {<json schema>}, ...}`

`load_schemas()` normalizes all of these to `{tool name: parameter schema}`.

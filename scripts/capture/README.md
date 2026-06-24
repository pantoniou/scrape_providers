# Capturing agent tool schemas

These scripts run a coding agent through a local [mitmproxy](https://mitmproxy.org)
and save the `tools` array it sends to the model API into
`src/scrape_providers/agent_schemas/<agent>.json`, where
`scrape-providers --agent-tool-schema <agent>[/<tool>]` can read it.

They use mitmproxy in **reverse-proxy** mode and point the agent's API **base URL**
at it (e.g. `ANTHROPIC_BASE_URL`). This is reliable because the agents' HTTP
clients (Node `fetch`/undici) *ignore* `HTTPS_PROXY`, but their SDKs honor the
base-URL env var. Plain HTTP to localhost means no CA trust is needed.

## Prerequisites

- `mitmproxy` — `pipx install mitmproxy`
- the agent CLI (`claude`, `codex`, `opencode`) on your `PATH`

## Usage

```bash
scripts/capture/capture-claude-code.sh     # -> agent_schemas/claude_code.json
scripts/capture/capture-codex.sh           # -> agent_schemas/codex.json (API-key mode)
scripts/capture/capture-opencode.sh        # -> agent_schemas/opencode.json
```

Each starts the proxy, sets the base-URL env var, launches the agent, and tears
the proxy down on exit. **Do one trivial turn** ("hi"), then quit the agent.

- Override the command/args after the script name (e.g. `./capture-codex.sh codex --model gpt-5.5-codex`).
- `MITM_PORT` changes the listen port.
- `CAPTURE_DEBUG=1` logs every host/POST so you can see traffic.

## Per-agent notes

- **Claude Code** → `ANTHROPIC_BASE_URL`, upstream `api.anthropic.com`. Works directly.
- **Codex** → works under **ChatGPT login** too. Codex's `OPENAI_BASE_URL` only
  applies in API-key mode, so the script instead overrides
  `model_providers.openai.base_url` via `-c` (which wins in any auth mode) and
  reverse-proxies to the ChatGPT backend (`chatgpt.com/backend-api/codex`). The
  outgoing request carries `tools`, so it's captured even if the upstream rejects
  it. For API-key mode: `CAPTURE_UPSTREAM=https://api.openai.com/v1 ...capture-codex.sh`.
- **opencode** is provider-agnostic; defaults to Anthropic. For an OpenAI-family
  model:
  ```bash
  CAPTURE_UPSTREAM=https://api.openai.com CAPTURE_BASE_ENV=OPENAI_BASE_URL \
  CAPTURE_BASE_PATH=/v1 scripts/capture/capture-opencode.sh
  ```

**Redact** prompts/secrets from anything you capture before committing (the
`tools` array itself carries no secrets).

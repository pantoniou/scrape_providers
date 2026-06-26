# scrape-providers

Scrape AI providers (OpenAI, Anthropic, Google, …) for their model endpoints,
capabilities, and pricing, and emit a consolidated YAML catalog.

## Install

```bash
pip install -e '.[dev]'
```

## Usage

```bash
# Scrape all registered providers and write the catalog to stdout
scrape-providers

# Scrape specific providers and write to a file
scrape-providers --provider anthropic --provider openai --output catalog.yaml

# Render a human-readable Markdown table instead of YAML
scrape-providers --format markdown

# Cut the noise: only curated latest/flagship models (see curation.py)
scrape-providers --curated -f markdown

# List the model ids a single provider serves
scrape-providers --list-provider-models openrouter

# Agent harnesses: list them, or list a harness's standard function-calling tools
scrape-providers --list-agents               # harness names
scrape-providers --list-agent-tools          # all harnesses + their tools
scrape-providers --list-agent-tools codex    # one harness

# Vendored tool JSON schemas (drop a captured tools payload in agent_schemas/)
scrape-providers --agent-tool-schema codex         # all of codex's tool schemas
scrape-providers --agent-tool-schema codex/shell   # one tool's schema

# Vendored system prompts (captured alongside the tools)
scrape-providers --agent-system-prompt claude_code

# Output just one provider's single model as PROVIDER/MODEL (split on first '/',
# so the model id may itself contain slashes)
scrape-providers --show deepseek/deepseek-v4-pro
scrape-providers --show openrouter/z-ai/glm-5.2 -f markdown

# Show / change the curated list (persisted to a config file)
scrape-providers --print-curated > my-curated.yaml
scrape-providers --set-curated my-curated.yaml

# LMArena Elo / rank is included by default; skip the fetch with --no-arena
scrape-providers --curated -f markdown
scrape-providers --curated --no-arena -f markdown

# The `agents` section (harness developer, system prompt, tool schemas) is on by
# default; drop it (and the per-model agent tags) with --no-agents
scrape-providers --no-agents

# Validate the output against the JSON schema before emitting
scrape-providers --validate

# Validate an existing catalog file (e.g. one a user edited) against the schema
scrape-providers --validate-file catalog.yaml

# Print the catalog JSON schema
scrape-providers --schema
```

## Agent tool schemas

`--list-agent-tools` shows the *names* of the standard function-calling tools each
agent harness (Codex, Claude Code, opencode) exposes. The full JSON parameter
schemas aren't published anywhere, but every request an agent sends to the model
API carries them in its `tools` array — so capture them with the bundled scripts:

```bash
scripts/capture/capture-claude-code.sh   # -> src/scrape_providers/agent_schemas/claude_code.json
scripts/capture/capture-codex.sh         # -> .../codex.json
scripts/capture/capture-opencode.sh      # -> .../opencode.json
```

Each starts a local [mitmproxy](https://mitmproxy.org) (`pipx install mitmproxy`),
launches the agent through it, and writes the captured `tools` array — plus the
request's **system prompt** to `src/scrape_providers/agent_prompts/<agent>.txt`.
Do one trivial turn, then quit the agent. Read them back with:

```bash
scrape-providers --agent-tool-schema codex                 # all tools (description + schema)
scrape-providers --agent-tool-schema codex/exec_command    # one tool
scrape-providers --agent-system-prompt codex               # the captured system prompt
```

See `scripts/capture/README.md` for details (custom commands, ports, redaction).

## Development

```bash
pytest          # run tests
ruff check .    # lint
ruff format .   # format
```

## Providers

| Provider | Source | Notes |
|---|---|---|
| `anthropic` | `/v1/models` API + pricing page | requires `ANTHROPIC_API_KEY` |
| `openrouter` | public `/api/v1/models` | no key needed; full catalog with pricing |
| `openai` | OpenRouter (`openai/*`) + native pricing page | characteristics via OpenRouter; pricing scraped from OpenAI's docs, OpenRouter as fallback |
| `deepseek` | native `/models` API + native pricing page | requires `DEEPSEEK_API_KEY`; cache-hit/miss pricing for all served models |

API keys are read from the environment. A convenient pattern:

```bash
set -a; . ~/work/fyai/providers.env; set +a
scrape-providers
```

## Output schema

The YAML catalog has three top-level sections:

- `models` — a list of each model's intrinsic capabilities; each entry's `name`
  is its **canonical id** (plus display name, context window, max output,
  modalities, capabilities, `open_source`, arena Elo/rank, and `agents` — the
  canonical agent harness(es) that natively drive it, e.g. gpt → `codex`).
- `agents` — each known coding-agent harness, fully described: `developer` (the
  company that builds it), `native_provider` (the catalog provider it natively
  targets; omitted for model-agnostic harnesses like opencode), `protocol`, the
  vendored `system_prompt`, and the vendored `tools` (tool name → schema). On by
  default; `--no-agents` omits this section *and* the per-model `agents` tags.
- `providers` — each provider has a `root_url` (API host root), an `endpoints`
  list of `{protocol, endpoint, tools}` entries (a provider may expose its models
  over several protocols — `chat_completions`, `responses`, `messages` — e.g.
  OpenAI via chat completions *and* responses, DeepSeek via chat completions *and*
  an Anthropic-format messages endpoint; each endpoint splits its built-in tools
  into `hosted_tools` (run on the provider, e.g. `web_search`, `code_interpreter`)
  and `local_tools` (the caller executes, e.g. `local_shell`, `computer_use`)),
  and the models it serves. Each offering has
  `canonical_id` (the key into `models`), `provider_model_id` (the id that provider
  uses), and `pricing`. A full request URL is `root_url + endpoints[i].endpoint`.

A model served by several providers appears once under `models`. The canonical id
drops vendor prefixes and lowercases (so OpenAI's `gpt-5.5` and OpenRouter's
`openai/gpt-5.5` collapse to `gpt-5.5`); `canonical.py` holds an `ALIASES`
override table for ids the rule mishandles.

## Architecture

- `src/scrape_providers/models.py` — normalized data model (provider, model,
  pricing) that every scraper maps into. The YAML output serializes this model,
  not raw provider responses.
- `src/scrape_providers/base.py` — the `Scraper` interface each provider
  implements.
- `src/scrape_providers/providers/` — one module per provider; quirks (auth,
  HTML vs JSON, pagination) stay isolated here.
- `src/scrape_providers/registry.py` — maps provider names to scraper classes.
- `src/scrape_providers/emit.py` — deterministic YAML emission (stable key
  ordering so run-to-run diffs are meaningful).
- `src/scrape_providers/cli.py` — command-line entry point.

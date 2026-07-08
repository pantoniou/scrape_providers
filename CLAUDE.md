# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python tool that scrapes AI providers for their model endpoints, capabilities,
and pricing, and emits a consolidated catalog as YAML or Markdown.

## Commands

```bash
pip install -e '.[dev]'          # install (src layout, editable)

scrape-providers                 # scrape all providers -> YAML on stdout
scrape-providers --list-providers  # list registered provider names
scrape-providers -f markdown     # render Markdown tables instead
scrape-providers --provider openrouter -o catalog.yaml

pytest                           # run tests
pytest -k openrouter             # single test / pattern
ruff check . && ruff format .    # lint + format
```

### API keys

Keys are read from the environment. They live in `~/work/fyai/providers.env`
(`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`):

```bash
set -a; . ~/work/fyai/providers.env; set +a
```

Live network tests (e.g. `test_anthropic_live_scrape`) skip themselves when the
relevant key is absent, so `pytest` passes offline. The OpenRouter-backed
scrapers need no key.

## Architecture

Scraping (fetch + parse) is kept separate from the normalized model and from
output formatting, so a change in one layer doesn't ripple into the others.

- `models.py` — normalized pydantic model (`Provider` → `Model` → `Pricing`).
  **The catalog output is a serialization of these types, not of raw provider
  responses.** Pricing is normalized to USD per million tokens.
- `base.py` — `Scraper` ABC: each provider's `scrape()` returns a fully
  normalized `Provider`. Holds a shared httpx client; usable as a context manager.
- `providers/` — one module per provider. Provider quirks (auth, JSON vs HTML,
  pagination, prefix handling) stay isolated here.
- `registry.py` — maps provider name → scraper class (`available()`, `get()`).
- `emit.py` — `build_catalog` splits the scraped data into three top-level
  sections: `models` (a list of intrinsic capabilities, each entry's `name` is its
  canonical id — context, modalities, capabilities, open_source, arena, plus
  `agents`: the canonical agent harness(es) that natively drive it), `providers`
  (each with `root_url`, an `endpoints` list of {protocol, endpoint}, and
  offerings of `canonical_id`/`provider_model_id`/`pricing`), and `agents` (built
  from `agent_profiles.build_agents()` — see below). A model is tagged with agent
  A when its native provider serves it (bare id) or its id carries A's
  `native_provider` as a vendor prefix (OpenRouter's `openai/…`). `build_catalog`
  takes `include_agents=True`; `--no-agents` threads `include_agents=False`
  through `pruned_catalog`/`to_yaml`/`to_markdown` to drop both the section and
  the per-model tags. `to_yaml` emits with intentional field order
  (`sort_keys=False`, models list sorted by name for determinism), pruning
  None/empty; `to_markdown` is the human view (agents shown as tool names, not
  full schemas).
- `canonical.py` — `canonical_id` maps a provider-specific model id to a shared
  key (drops vendor prefix + lowercases; `ALIASES` overrides), so the same model
  served by multiple providers collapses into one `models` entry.
- `tools.py` — curated map of built-in tools per `(provider, protocol)`, split
  into `hosted` (run on the provider: web_search, code_interpreter, …) and `local`
  (the caller executes: `local_shell`/bash, computer_use, function_calling, …). No
  provider API enumerates these, so they're hand-maintained and attached to the
  provider's endpoints (the same model exposes different tools under different
  protocols). Scrapers build each surface via `endpoint_for(provider, protocol,
  path)`, which populates `Endpoint.hosted_tools` / `local_tools`.
- `agent_profiles.py` — curated tool/function sets of well-known agent harnesses
  (Codex: shell/apply_patch/update_plan; Claude Code: Bash/Read/Edit/…). These are
  harness-level and model-agnostic (any function-calling model can be driven by
  them), so they're separate from provider data. Each profile also carries a
  `developer` (the company that builds it) and a `provider` (the native catalog
  provider it targets, `None` for model-agnostic harnesses); `build_agents()`
  assembles the full catalog `agents` section (name, developer, native_provider,
  protocol, system_prompt, tools) and `native_provider_agents()` maps a provider
  key → agent name(s) for the per-model tagging in `emit`. CLI:
  `--list-agent-tools [AGENT]`,
  which derives tool names from a vendored capture when present (`tool_names()` /
  `has_capture()`), falling back to the curated `tools` list otherwise.
  Full tool JSON schemas aren't scrapable (Rust/Zod source, or closed) — instead
  they're vendored: drop a captured `tools` payload into `agent_schemas/<agent>.json`
  (capture via a proxy like ccglass/mitmproxy; see that dir's README), and
  `load_schemas`/`index_tools` normalize Anthropic/OpenAI shapes. CLI:
  `--agent-tool-schema AGENT[/TOOL]`. The same capture also saves each agent's
  **system prompt** to `agent_prompts/<agent>.txt` (the addon's
  `extract_system_prompt` pulls it from the Anthropic `system` field, the OpenAI
  Responses `instructions` field, or leading `system`/`developer` turns);
  `system_prompt()`/`has_system_prompt()` read it. CLI: `--agent-system-prompt AGENT`.
- `curation.py` — the `--curated` allowlist: `DEFAULT_CURATED` built-ins plus an
  optional override config (`load_curated`/`save_curated`) at
  `$SCRAPE_PROVIDERS_CURATED` or `$XDG_CONFIG_HOME/scrape-providers/curated.yaml`.
  CLI: `--print-curated` shows the active list, `--set-curated FILE` persists one.
- `schema.py` + `catalog.schema.json` — JSON Schema (draft 2020-12) for the YAML
  output and `validate_catalog`. CLI: `--validate` checks the built catalog,
  `--schema` prints the schema. The schema is strict (prices `minimum: 0`,
  protocol enum, `additionalProperties: false`) so it catches bad scraped data.
- `cli.py` — argparse entry point (`--provider`, `--format`, `--output`,
  `--list-providers`, `--show`, `--curated`, `--arena`, `--validate`).

### Provider sourcing (important)

OpenRouter's public `/api/v1/models` is the data source for **three** scrapers,
because it returns pricing + capabilities in one unauthenticated call:

- `openrouter` — the full routed catalog.
- `openai` — filters OpenRouter by `openai/` prefix for characteristics (the
  native `/v1/models` API returns only bare ids), but **pricing is scraped
  natively** from `platform.openai.com/docs/pricing` (`parse_openai_pricing`) and
  joined by id, with OpenRouter pricing as fallback for models not on the page
  (the page only prices currently-offered models, so ~8 of 62 match natively).

`deepseek` is **fully native** (no OpenRouter): the `/models` API gives the
served ids and the docs pricing page (a transposed table parsed in
`providers/deepseek.py`) gives cache-hit/miss input, output, context, and max
output for every model. DeepSeek caching is automatic, so `cache_read` pricing is
available for all its models — the reason to scrape it directly rather than rely
on OpenRouter's spotty per-host cache figures. Note this exposes only the models
DeepSeek itself serves (currently 2), not the third-party-hosted `deepseek/*`
variants on OpenRouter.

`google` follows the `openai` pattern (no key needed): OpenRouter's `google/*`
entries give the served ids and characteristics, and pricing is scraped
natively from `ai.google.dev/gemini-api/docs/pricing`. That page has no
per-row model id — each model is an `<h2>` heading followed by one or more
pricing tables (Standard/Batch/Flex/Priority tiers under their own `<h3>`) — so
`parse_gemini_pricing` walks the document in order and joins on a slug derived
from the heading text (stripping parenthetical nicknames like "(Nano Banana
2)" and emoji), matched against the OpenRouter id. Only the Standard/untiered
table is used; Batch/Flex/Priority are discounted/premium variants of the same
figures. Google exposes two real surfaces — the native `generateContent` REST
API and an OpenAI-compatible `chat/completions` endpoint — both listed under
`endpoints`; `generate_content` was added to the protocol enum for the former.

Arena annotation (`arena.py`) is on by default (`--no-arena` skips the fetch) and
adds LMArena Elo/rank to each model. The
leaderboard has no public JSON API, so the data is extracted from escaped JSON in
the Next.js page's RSC payload; the `overall` board is the first contiguous run of
increasing ranks. Scores join onto models by **exact** normalized name only —
fuzzy matching is avoided so e.g. `gpt-5.5-pro` never inherits `gpt-5.5`'s Elo.

`anthropic` is the exception: it uses the native paginated `/v1/models` API for
characteristics, joined to the public pricing-page HTML table (parsed with
selectolax) on normalized display name. Pricing there is best-effort — a model
still emits without a pricing block if the page can't be parsed.

When adding a provider with its own pricing API/page, prefer the native source
and add a dedicated scraper rather than extending the OpenRouter filter pattern.

## Conventions

Use current Claude model IDs when referencing Anthropic models: Opus 4.8
(`claude-opus-4-8`), Sonnet 4.6 (`claude-sonnet-4-6`), Haiku 4.5
(`claude-haiku-4-5-20251001`), Fable 5 (`claude-fable-5`). Verify provider model
IDs against the live source rather than hardcoding from memory.

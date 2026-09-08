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
(`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`,
`FIREWORKS_API_KEY`, `OPENCODE_API_KEY`, `META_API_KEY`):

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
  canonical id — context, modalities, capabilities (both in the controlled
  vocabulary of `capabilities.py`), open_source, arena, `introduced`/`updated`
  (see below), plus
  `agents`: the canonical agent harness(es) that natively drive it), `providers`
  (each with `root_url`, an `endpoints` list of {protocol, endpoint}, and
  offerings of `canonical_id`/`provider_model_id`/`pricing` plus
  `reported_modalities`/`reported_capabilities` — what *that* provider publishes
  for the model, as against the merged entry under `models`; **reported, not
  supported**, since providers only ever publish a positive set in their own
  vocabulary (OpenRouter lists request parameters, Anthropic lists features, Zen
  publishes nothing), so an absent entry is never evidence the route lacks it),
  and `agents` (built
  from `agent_profiles.build_agents()` — see below). When several providers serve
  the same canonical model, `_merge_capabilities` builds one entry from all of
  their views. A first-party provider serves only models it built, so if one of
  them is present **its description is the model** and the resellers' claims are
  dropped from the intrinsic entry — a reseller can't route a capability the
  vendor doesn't have, and where they disagree the vendor is right (Anthropic's
  documented 1M context beats OpenRouter's 1048576). With no first-party provider
  (open-weight models nobody in the catalog built) every reseller counts and the
  strongest claim wins: widest context window / max output, union of modalities
  and capabilities. `open_source` is an OR across *all* views including the
  vendor's, because a provider that never mentions weights leaves the field at its
  `False` default — an absent claim, not a claim of closedness. `display_name`
  and `arena` hold a single value rather than a maximum, so `_naming_rank` picks
  one: a name that keeps every parameter-count moniker in the canonical id
  (`gpt-oss-120b`) beats one that drops it, because for a family differing only by
  size a name without the size doesn't identify the model; distance from the
  vendor breaks the tie (`_RESELLER_RANK`: first-party providers serve only models
  they built and rank first, then fireworks, opencode, openrouter — the last
  prefixes a vendor label the vendor doesn't use).
  A model is tagged with agent
  A when its native provider serves it (bare id) or its id carries A's
  `native_provider` as a vendor prefix (OpenRouter's `openai/…`). `build_catalog`
  takes `include_agents=True`; `--no-agents` threads `include_agents=False`
  through `pruned_catalog`/`to_yaml`/`to_markdown` to drop both the section and
  the per-model tags. `to_yaml` emits with intentional field order
  (`sort_keys=False`, models list sorted by name for determinism), pruning
  None/empty; `to_markdown` is the human view (agents shown as tool names, not
  full schemas). `build_catalog` also warns on stderr, per provider, about served
  models with no input price. Such a model is **kept** (the route is real and
  callable, and an absent `pricing` block says so honestly) — but never silently,
  since the cause is usually that the provider delisted the model or moved the
  page its price is scraped from. This is the catalog's only signal that a
  pricing scraper has silently gone stale.
- `personas.py` — per-provider **persona rankings**: `providers[].personas` maps
  an orchestration persona (`orchestrate`, `plan`, `code`, `explore`, `review`,
  `summarize` — a fixed vocabulary, emitted in that order) to that provider's own
  models, best first, as `provider_model_id`s so a list is dispatchable as-is.
  The question is "which of *this* lineup runs this step", so the answer is
  per-provider and differs per step. **Strength** is the merged entry's LMArena
  Elo — taken from `models`, not the offering, so every provider serving a scored
  model inherits it (Fireworks' `glm-5p3` is the `glm-5.3` the board ranks).
  Where the board scores nothing directly, `resolve_strength` works outwards from
  real relationships rather than guessing: the Elo of the model an offering is a
  *variant* of (`_VARIANT_SUFFIX` — `-contributor` and `:free` buy a discount on
  identical weights, `-pro`/`-max`/`-latest` are the same weights turned up or
  aliased), then the strongest scored member of its **family**, which for a newer
  version of a scored line is a floor rather than a guess (no vendor ships a 1.3
  weaker than the 1.2 it replaces). Parameter counts are protected, so
  `gpt-oss-20b` never inherits `gpt-oss-120b`, and families are tight enough that
  `claude-haiku-4.5` never inherits Opus. **Cost** is a blend of that
  provider's input and output prices, weighted per persona (`input_share`),
  because personas have different token shapes — a reviewer reads far more than
  it writes, a planner emits long reasoning. Each persona then scores
  `strength_weight` against cost, so `orchestrate` (0.90) takes the strongest and
  barely looks at the bill while `summarize` (0.15) takes the cheapest competent
  model. **Every priced model is ranked**, never dropped: the question is which
  of the lineup runs a step, so a weaker or dearer model is tiered lower and the
  last resort stays nameable. Models whose strength is evidenced rank above those
  where only price is known, and that tail is ordered by price alone — which,
  used as a stand-in for both axes, means dearest first for a strength-led
  persona and cheapest first for a cost-led one. Only an offering with no usable
  price is left out. Because the Elo comes from the merged entries, a
  `--select providers/<name>/personas` deliberately scrapes **every** provider,
  unlike other `providers/<name>/…` paths.
- `capabilities.py` — the controlled vocabulary for `models[].modalities` and
  `models[].capabilities`. Providers describe the same abilities in
  incomparable words — OpenRouter and the scrapers built on it (`openai`,
  `google`) publish *request parameters* (`tools`, but also `top_a`, `min_p`),
  Anthropic publishes product features (`thinking`, `citations`), DeepSeek
  mixes in a wire protocol (`anthropic_api`) — so raw strings can't be compared
  across providers. Two rules draw the line: a **capability** is something the
  model can do that changes what you can build (a knob that tunes an ability
  every model already has is not one: `temperature`, `max_tokens`, `tool_choice`,
  which `tool_calling` already implies), and it is a property of the **model**,
  not the route (`anthropic_api` belongs on the provider's endpoints). Three
  tables: `_MODALITY_ALIASES`/`_CAPABILITY_ALIASES` map provider terms onto
  `MODALITIES`/`CAPABILITIES` (collapsing `tools`/`tool_calls`/`function_calling`
  into `tool_calling`, `thinking`/`include_reasoning` into `reasoning`, `file`
  into `pdf`), and `_IGNORED` lists terms deliberately dropped — so an
  unclassified term can be told apart from one already judged not to be a
  capability, and `build_catalog` warns on stderr about anything in neither.
  `structured_outputs` (schema-guaranteed) stays separate from `json_mode`
  (valid JSON only), since the former is strictly stronger. Emission follows the
  declared order of the tuples, not input or alphabetical order. **The
  vocabulary applies only to the merged entries under `models`**; each
  offering's `reported_*` keeps its provider's own words verbatim, which is the
  evidence those fields exist to preserve. The schema mirrors both tuples as
  `enum`s and a test pins them together so they can't drift.
- `canonical.py` — `canonical_id` maps a provider-specific model id to a shared
  key (drops vendor prefix + lowercases, then normalizes provider-mangled version
  numbers — see below; `ALIASES` overrides), so the same model served by multiple
  providers collapses into one `models` entry.
- `tools.py` — curated map of built-in tools and behavioral capabilities per
  `(provider, protocol)`, with tools split into `hosted` (run on the provider:
  web_search, code_interpreter, …) and `local` (the caller executes:
  `local_shell`/bash, computer_use, function_calling, …). No provider API
  enumerates all of these, so they're hand-maintained and attached to the
  provider's endpoints (the same model exposes different behavior and tools
  under different protocols). Scrapers build each surface via
  `endpoint_for(provider, protocol, path)`, which populates
  `Endpoint.hosted_tools`, `local_tools`, and `capabilities`.
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
  protocol enum, `additionalProperties: false`) so it catches bad scraped data,
  and `validate_catalog` passes an explicit `jsonschema.FormatChecker()` because
  `format` is annotation-only without one — a `format: date` field would
  otherwise accept `28/05/2026`.
  Every property carries a `description`; the schema is the reference for what a
  field means, so keep them there rather than only in this file.
- `cli.py` — argparse entry point (`--provider`, `--format`, `--output`,
  `--list-providers`, `--show`, `--curated`, `--arena`, `--validate`,
  `--select`).

### Partial output (`--select PATH`)

`--select` emits just the part of the catalog at a slash-separated path —
`models`, `models/gpt-5.5`, `providers/openrouter/endpoints`, `agents/codex`,
`agents/codex/system_prompt`. `emit.select_path` walks the pruned catalog:
mappings are indexed by key, and **lists by their entries' `name`, never by
position**, since the catalog's lists are named collections whose order is an
emission detail. Empty segments are ignored, so `models` and `models/` are one
path and `""` is the whole catalog. A failed segment raises `KeyError` naming
where it failed and listing the alternatives.

A leaf that isn't a dict or list prints verbatim (a system prompt stays a system
prompt, not a quoted YAML scalar); everything else prints as YAML. `--select` is
therefore incompatible with `-f markdown`, which renders whole catalogs only.

The CLI also scrapes only what the path needs: `agents/…` is curated data and
scrapes nothing at all, `providers/<name>/…` scrapes that one provider, and
`models/…` still needs every provider because a model's entry is merged from all
of them. An explicit `--provider` always wins over this narrowing.

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
on OpenRouter's spotty per-host cache figures. **Currently DeepSeek emits no
pricing at all**: as of 2026-09-07 the English pricing page's content is missing
upstream (still canonical in their sitemap and still routed by their JS bundle,
but the chunk isn't deployed, and their server returns the SPA index for missing
paths instead of a 404 — hence a useless HTTP 200). The only live page is the
`zh-cn` locale, in CNY and now peak/off-peak tiered; converting it would make the
catalog's USD figures FX-derived, so this is deliberately left unfixed and
surfaced by the unpriced-models warning instead. If the page returns, take the
**peak (standard)** tier, as the Google scraper does with Standard. Note this exposes only the models
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

`fireworks` is **fully native** (`FIREWORKS_API_KEY`): the OpenAI-compatible
`/inference/v1/models` API gives the served ids plus context length and
tool/image-input flags, and per-token pricing comes from each model's public page
(`fireworks.ai/models/fireworks/<slug>`) — its "Available Serverless" block holds
the input / cached input / output triple and the display name, and a
huggingface.co link on the page is what marks the model open-weight. Fireworks'
`/pricing` page covers only fine-tuning and dedicated deployments, which is why
pricing is scraped one model page at a time. The label above the prices names the
component order, so it (not position) drives the parse; embedding/reranking models
publish a single unlabelled price, read as `input`. Fireworks **routers**
(`accounts/fireworks/routers/…`, the `-fast`/`-turbo` speed tiers) have no public
page and are emitted without pricing rather than inheriting the base model's.
Three surfaces: `chat_completions`, `responses`, and an Anthropic-format
`messages` endpoint.

`opencode` is OpenCode Zen (sst's gateway), also **fully native**
(`OPENCODE_API_KEY`): `/zen/v1/models` gives the served ids and the docs page
(`opencode.ai/docs/zen`) supplies a models table (display name, id, per-model
endpoint) and a pricing table. The pricing table is keyed by display name, so it's
joined to ids through the models table. Two wrinkles: long-context models are
priced in `(≤ N tokens)` / `(> N tokens)` tiers and only the base tier is kept, and
free models list `Free`, emitted as a real 0.0 rather than missing pricing. Zen
forwards each model to its vendor, so all four protocols are listed under
`endpoints` (`responses` for GPT, `messages` for Claude, `generate_content` for
Gemini, plus OpenAI-compatible `chat_completions`) and vendor-dependent endpoint
capabilities are left unstated. The docs give no context window, so Zen models
carry one only when another provider serves the same canonical model.

`meta` is **fully native** (`META_API_KEY`): Meta's Model API
(`api.meta.ai`/`dev.meta.ai`) is a separate, billed product from the
open-weight Llama releases, serving Meta's proprietary Muse Spark family
(`muse-spark-1.1`/`1.2`/`1.3`, plus discounted `-contributor` variants). The
`GET /v1/models` API gives served ids but no context window or modality info
(the docs point elsewhere for that), so those — plus pricing — come from two
docs pages fetched as their raw **Markdown source** (`<url>.md`, not rendered
HTML — the docs site is client-rendered, and the `.md` form is far simpler and
more stable to parse): `docs/models` (a table of context window + input
modalities per model id) and `docs/pricing-rate-limits` (per-tier pricing,
where each tier's section states its member model ids in a "Models: `id`,
`id`." sentence ahead of one shared price table, rather than a price per row).
Muse Spark's capabilities (tool calling, structured outputs, always-on
reasoning with `reasoning_effort`, web search grounding) are fixed across every
version and described only in prose, so they're hardcoded rather than parsed.
Meta also serves Muse Image (image generation) and Muse Voice Transcribe
(speech-to-text) on the same API, but neither fits this catalog's
per-million-token chat-model shape (priced per-image / per-audio-hour), so only
Muse Spark is scraped.

Two providers mangle version numbers in their ids, so `canonical.py` normalizes
both. Fireworks writes the decimal point as `p` (`kimi-k2p6`, `glm-5p2`); a
digit-`p`-digit run is rewritten back to a decimal point, and no other provider's
ids contain that pattern. Anthropic writes it as a dash and serves dated snapshot
ids for older generations (`claude-opus-4-8`, `claude-opus-4-5-20251101`), so for
`claude-` ids a trailing `-YYYYMMDD` is dropped and digit-dash-digit becomes a
decimal point — collapsing them with OpenRouter's `anthropic/claude-opus-4.8` and
Zen's `claude-opus-4-8`. Both rewrites are deliberately scoped (the Anthropic one
by the `claude-` prefix) so nothing else loses a dash or a numeric suffix.

### Model dates

`introduced` and `updated` are properties of the **model**, not of any provider's
listing of it, so they appear only on the merged `models` entries — never on an
offering. Scrapers still report them per provider, as the evidence `_merge_dates`
distils:

- `introduced` is the release date. A provider's `created` stamp is only the date
  *it* began serving the model, so the vendor's is the release date itself while a
  reseller's bounds it from above (measured across the catalog, the first-party
  date is earlier than or equal to the resellers' in 51 of 54 cases). The vendor
  wins where the catalog scrapes that vendor — the `_merge_capabilities` rule
  again — else the earliest reseller date is taken as the best bound. Coverage is
  ~97%.
- `updated` is the date the model behind a *stable* id was last revised in place.
  It is rare by construction: a new version normally mints a new canonical id
  (`gemini-3.7-flash` → `gemini-3.8-flash`) and therefore a new entry rather than
  an update, so only a provider publishing a version marker for an unchanging id
  can fill it in. DeepSeek is currently the only one (`parse_versions` reads the
  docs' model-version row: `deepseek-v4-flash` runs `DeepSeek-V4-Flash-0731`).
  The build suffix carries no year, so it resolves to the most recent such
  month/day not in the future.

Two providers ship a `created` field without meaning it, and `created_date` drops
both rather than emit a bogus date: **OpenCode Zen** stamps every model with the
time of the request, and **Meta** returns a literal `0`. Two more are
OpenRouter-derived yet rank first-party (`openai`, `google`), where passing
OpenRouter's listing date along would disguise a reseller bound as the vendor's
word: `openai` therefore takes dates from its native `/v1/models` (the one thing
that endpoint is good for) when a key is present, and `google` — which publishes
no native model list — reports none, falling back to the reseller bound.

Arena annotation (`arena.py`) is on by default (`--no-arena` skips the fetch) and
adds LMArena Elo/rank to each model. The
leaderboard has no public JSON API, so the data is extracted from escaped JSON in
the Next.js page's RSC payload; the `overall` board is the first contiguous run of
increasing ranks. Scores join onto models by **exact** normalized name, with one
narrow exception: the board scores reasoning-effort *variants* as their own rows
(`claude-opus-4-8-high`, `gpt-5.6-sol-xhigh`, `gemini-3-flash (thinking-minimal)`,
`claude-sonnet-4-5-20250929-high-32k`), which are the same model at a different
effort rather than a different model. `_EFFORT_SUFFIX` — a **closed vocabulary**
of effort words (`high`/`xhigh`/`max`/`medium`/`low`/`minimal`/`thinking`, plus an
optional context qualifier and the parenthesized form), never a general suffix
pattern — strips them so the row falls back onto the base model. An exact row
always wins over a stripped one, so a variant only ever fills a model the board
doesn't otherwise score; where several variants collapse onto one unscored model
the strongest is kept. Fuzzy matching beyond that is avoided so e.g.
`gpt-5.5-pro` never inherits `gpt-5.5`'s Elo — the list is enumerated precisely
because `-pro` names a different model while `-high` names a knob.

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

import os

import pytest
import yaml

from scrape_providers import registry
from scrape_providers.cli import main
from scrape_providers.emit import to_markdown, to_yaml
from scrape_providers.models import Endpoint, Model, Pricing, Provider
from scrape_providers.providers.anthropic import _norm, _parse_pricing_table


def test_registry_lists_anthropic():
    assert "anthropic" in registry.available()


def test_emit_is_deterministic_and_prunes_none():
    provider = Provider(
        name="demo",
        endpoints=[Endpoint(protocol="chat_completions", endpoint="/x")],
        models=[Model(id="m1", display_name="M1", pricing=Pricing(input=1.0))],
    )
    out = to_yaml([provider])
    assert to_yaml([provider]) == out  # deterministic
    parsed = yaml.safe_load(out)
    # capabilities live under top-level `models` as a list, name = canonical id
    m1_caps = next(m for m in parsed["models"] if m["name"] == "m1")
    assert m1_caps["display_name"] == "M1"
    # provider entry references the model and carries pricing
    provider = parsed["providers"][0]
    offering = provider["models"][0]
    assert provider["name"] == "demo"
    # protocol/endpoint are hoisted to the provider's endpoints list
    assert provider["endpoints"][0]["protocol"] == "chat_completions"
    assert offering["canonical_id"] == "m1"
    assert offering["provider_model_id"] == "m1"
    assert "protocol" not in offering
    assert "output" not in offering["pricing"]  # None was pruned
    # intrinsic capability data is NOT duplicated into the provider offering
    assert "modalities" not in offering


def test_canonical_collapse_across_providers():
    from scrape_providers.emit import build_catalog

    # same logical model, different ids per provider
    providers = [
        Provider(name="openai", models=[Model(id="gpt-5.5")]),
        Provider(name="openrouter", models=[Model(id="openai/gpt-5.5")]),
    ]
    catalog = build_catalog(providers)
    # collapsed into a single models entry under the canonical id
    assert [m["name"] for m in catalog["models"]] == ["gpt-5.5"]
    # each provider keeps its own id but references the shared canonical id
    openai_off = catalog["providers"][0]["models"][0]
    openrouter_off = catalog["providers"][1]["models"][0]
    assert openai_off["provider_model_id"] == "gpt-5.5"
    assert openrouter_off["provider_model_id"] == "openai/gpt-5.5"
    assert openai_off["canonical_id"] == openrouter_off["canonical_id"] == "gpt-5.5"


def test_agents_section_and_model_annotation():
    from scrape_providers.emit import build_catalog

    providers = [
        Provider(name="openai", models=[Model(id="gpt-5.5")]),
        Provider(name="anthropic", models=[Model(id="claude-opus-4-8")]),
        Provider(
            name="openrouter",
            models=[Model(id="openai/gpt-5.5"), Model(id="z-ai/glm-5.2")],
        ),
    ]
    catalog = build_catalog(providers)

    # agents section: all known harnesses, fully described
    agents = {a["name"]: a for a in catalog["agents"]}
    assert {"codex", "claude_code", "opencode"} <= set(agents)
    assert agents["codex"]["developer"] == "OpenAI"
    assert agents["codex"]["native_provider"] == "openai"
    assert agents["claude_code"]["native_provider"] == "anthropic"
    # opencode is model-agnostic: no native provider
    assert agents["opencode"]["native_provider"] is None
    # vendored data is folded in (captures are committed)
    assert agents["codex"]["tools"] and agents["codex"]["system_prompt"]

    # models are tagged with the canonical agent that natively drives them
    by_name = {m["name"]: m for m in catalog["models"]}
    assert by_name["gpt-5.5"]["agents"] == ["codex"]  # native + openrouter prefix
    assert by_name["claude-opus-4-8"]["agents"] == ["claude_code"]
    assert by_name["glm-5.2"]["agents"] == []  # no native agent


def test_no_agents_omits_all_agent_output():
    from scrape_providers.emit import build_catalog

    providers = [Provider(name="openai", models=[Model(id="gpt-5.5")])]
    catalog = build_catalog(providers, include_agents=False)
    assert "agents" not in catalog
    assert catalog["models"][0]["agents"] == []  # pruned away on emit


def test_to_markdown():
    provider = Provider(
        name="demo",
        root_url="https://api.demo",
        endpoints=[Endpoint(protocol="messages", endpoint="/v1/messages")],
        models=[
            Model(
                id="m1",
                display_name="M1",
                context_window=1_000_000,
                max_output_tokens=64000,
                modalities=["text", "image"],
                pricing=Pricing(input=3.0, output=15.0),
            )
        ],
    )
    md = to_markdown([provider])
    assert "## demo" in md
    assert "| M1 |" in md
    assert "messages `/v1/messages`" in md  # endpoints shown in the provider meta line
    assert "1M" in md and "64K" in md
    assert "$3" in md and "$15" in md


def test_endpoint_tools_split_hosted_local():
    from scrape_providers.emit import build_catalog
    from scrape_providers.tools import endpoint_for

    provider = Provider(
        name="openai",
        root_url="https://api.openai.com",
        endpoints=[endpoint_for("openai", "responses", "/v1/responses")],
        models=[Model(id="gpt-5.5")],
    )
    endpoint = build_catalog([provider])["providers"][0]["endpoints"][0]
    assert "local_shell" in endpoint["local_tools"]  # the bash/shell tool, caller-run
    assert "web_search" in endpoint["hosted_tools"]  # runs on the provider
    assert "local_shell" not in endpoint["hosted_tools"]


def test_registry_lists_all_providers():
    assert set(registry.available()) >= {"anthropic", "deepseek", "google", "openai", "openrouter"}


def test_openrouter_negative_price_sentinel_dropped():
    from scrape_providers.providers import openrouter

    raw = {
        "id": "x/y",
        "pricing": {"prompt": "0.000001", "completion": "-1", "request": "-1"},
    }
    m = openrouter.to_model(raw)
    assert m.pricing.input == 1.0
    assert m.pricing.output is None  # negative sentinel dropped, not -1e6
    assert "per_request" not in m.pricing.extra


def test_openrouter_to_model():
    from scrape_providers.providers import openrouter

    raw = {
        "id": "openai/gpt-5.5",
        "name": "OpenAI: GPT-5.5",
        "context_length": 400000,
        "architecture": {"input_modalities": ["text", "image"]},
        "top_provider": {"max_completion_tokens": 128000},
        "supported_parameters": ["tools", "structured_outputs"],
        "pricing": {"prompt": "0.000005", "completion": "0.00003", "input_cache_read": "0.0000005"},
    }
    m = openrouter.to_model(raw, strip_prefix=True)
    assert m.id == "gpt-5.5"  # prefix stripped
    assert m.context_window == 400000
    assert m.max_output_tokens == 128000
    assert m.modalities == ["text", "image"]
    assert m.capabilities == ["structured_outputs", "tools"]
    # per-token -> per-million
    assert m.pricing.input == 5.0
    assert m.pricing.output == 30.0
    assert m.pricing.extra["cache_read"] == 0.5


def test_parse_openai_pricing_two_tier_and_modality():
    from scrape_providers.providers.openai import parse_openai_pricing

    html = """
    <table>
      <tr><th></th><th>Short context</th><th>Long context</th></tr>
      <tr><td>Model</td><td>Input</td><td>Cached input</td><td>Output</td>
          <td>Input</td><td>Cached input</td><td>Output</td></tr>
      <tr><td>gpt-5.5</td><td>$5.00</td><td>$0.50</td><td>$30.00</td>
          <td>$10.00</td><td>$1.00</td><td>$45.00</td></tr>
    </table>
    <table>
      <tr><th>Model</th><th>Modality</th><th>Input</th><th>Cached input</th><th>Output</th></tr>
      <tr><td>gpt-image-2</td><td>Image</td><td>$8.00</td><td>$2.00</td><td>$30.00</td></tr>
      <tr><td>Text</td><td>$2.50</td><td>$0.625</td><td>-</td><td></td></tr>
    </table>
    <table>
      <tr><th>Model</th><th>Training</th><th>Input</th><th>Cached input</th><th>Output</th></tr>
      <tr><td>o4-mini</td><td>$100 / hour</td><td>$4.00</td><td>$1.00</td><td>$16.00</td></tr>
    </table>
    """
    pricing = parse_openai_pricing(html)
    # two-tier table: takes short-context (standard) input/output + cached
    assert pricing["gpt-5.5"].input == 5.0
    assert pricing["gpt-5.5"].output == 30.0
    assert pricing["gpt-5.5"].extra["cache_read"] == 0.5
    # modality table parsed; the "Text" sub-row is not treated as a model
    assert pricing["gpt-image-2"].input == 8.0
    assert "Text" not in pricing
    # fine-tuning ("Training") table is skipped entirely
    assert "o4-mini" not in pricing


def test_parse_gemini_pricing_headline_tier_and_slugify():
    from scrape_providers.providers.google import parse_gemini_pricing

    html = """
    <h2>Gemini 3.1 Flash Image (Nano Banana 2) \U0001f34c</h2>
    <h3>Standard</h3>
    <table>
      <tr><th></th><th>Free Tier</th><th>Paid Tier, per 1M tokens in USD</th></tr>
      <tr><td>Input price</td><td>Not available</td><td>$0.50</td></tr>
      <tr><td>Output price (including thinking tokens)</td><td>Not available</td><td>$3.00</td></tr>
      <tr><td>Context caching price</td><td>Not available</td><td>$0.05</td></tr>
    </table>
    <h3>Batch</h3>
    <table>
      <tr><th></th><th>Free Tier</th><th>Paid Tier, per 1M tokens in USD</th></tr>
      <tr><td>Input price</td><td>Not available</td><td>$0.25</td></tr>
      <tr><td>Output price (including thinking tokens)</td><td>Not available</td><td>$1.50</td></tr>
    </table>
    <h2>Gemma 4</h2>
    <table>
      <tr><th></th><th>Free Tier</th><th>Paid Tier, per 1M tokens in USD</th></tr>
      <tr><td>Input price</td><td>Free of charge</td><td>Not available</td></tr>
      <tr><td>Output price</td><td>Free of charge</td><td>Not available</td></tr>
    </table>
    """
    pricing = parse_gemini_pricing(html)
    # heading's parenthetical nickname + emoji dropped; slug matches the OpenRouter id
    assert "gemini-3.1-flash-image" in pricing
    opus = pricing["gemini-3.1-flash-image"]
    # the Standard tier is the headline; Batch's discounted rate is ignored
    assert opus.input == 0.50
    assert opus.output == 3.00
    assert opus.extra["cache_read"] == 0.05
    # a model with no priced tier ("Not available" everywhere) emits nothing
    assert "gemma-4" not in pricing


def test_parse_gemini_pricing_skips_per_image_models():
    from scrape_providers.providers.google import parse_gemini_pricing

    html = """
    <h2>Gemini 2.5 Flash Image (Nano Banana) \U0001f34c</h2>
    <h3>Standard</h3>
    <table>
      <tr><th></th><th>Free Tier</th><th>Paid Tier, per 1M tokens in USD</th></tr>
      <tr><td>Input price</td><td>Not available</td><td>$0.30 (text / image)</td></tr>
      <tr><td>Output price</td><td>Not available</td><td>$0.039 per image*</td></tr>
    </table>
    """
    # the output is priced per image, not per token: the whole model is skipped
    # so the scraper falls back to OpenRouter's per-token rate (a partial
    # override would clobber the fallback's output figure)
    assert parse_gemini_pricing(html) == {}


def test_google_filters_openrouter_prefix_and_overrides_pricing(monkeypatch):
    from scrape_providers.providers import google

    raw_models = [
        {
            "id": "google/gemini-3.5-flash",
            "name": "Google: Gemini 3.5 Flash",
            "context_length": 1048576,
            "architecture": {"input_modalities": ["text", "image"]},
            "top_provider": {"max_completion_tokens": 65536},
            "supported_parameters": ["tools"],
            "pricing": {"prompt": "0.0000015", "completion": "0.000009"},
        },
        {"id": "openai/gpt-5.5", "pricing": {}},  # non-google: must be filtered out
    ]
    monkeypatch.setattr(google.openrouter, "fetch_models", lambda client: raw_models)
    scraper = google.GoogleScraper(client=object())
    scraper._fetch_native_pricing = lambda: {"gemini-3.5-flash": Pricing(input=1.0, output=5.0)}
    provider = scraper.scrape()
    assert [m.id for m in provider.models] == ["gemini-3.5-flash"]
    # native pricing override wins over OpenRouter's routed rate
    assert provider.models[0].pricing.input == 1.0
    assert provider.models[0].pricing.output == 5.0
    assert [e.protocol for e in provider.endpoints] == ["generate_content", "chat_completions"]


def test_curate_filters_and_orders():
    from scrape_providers.curation import CURATED, curate

    provider = Provider(
        name="anthropic",
        models=[
            Model(id="claude-sonnet-4-6"),
            Model(id="old-noise-model"),
            Model(id="claude-opus-4-8"),
        ],
    )
    curated = curate(provider)
    ids = [m.id for m in curated.models]
    # noise dropped; kept models follow the curated order, not scrape order
    assert "old-noise-model" not in ids
    assert ids == [i for i in CURATED["anthropic"] if i in {"claude-opus-4-8", "claude-sonnet-4-6"}]


def test_is_open_source():
    from scrape_providers.oss import is_open_source

    assert is_open_source("z-ai/glm-5.2")
    assert is_open_source("deepseek/deepseek-v4-pro")
    assert is_open_source("openai/gpt-oss-120b")  # name marker on a closed vendor
    assert is_open_source("google/gemma-3-27b")
    assert not is_open_source("openai/gpt-5.5")
    assert not is_open_source("anthropic/claude-opus-4-8")
    assert not is_open_source("claude-opus-4-8")  # no prefix, proprietary name


def test_tokens_formatting():
    from scrape_providers.emit import _tokens

    assert _tokens(1_000_000) == "1M"
    assert _tokens(1_050_000) == "1.05M"
    assert _tokens(1_048_576) == "1.05M"
    assert _tokens(65536) == "65.54K"
    assert _tokens(64000) == "64K"
    assert _tokens(None) == "—"


def test_anthropic_pricing_retries_until_plausible(monkeypatch):
    from scrape_providers.providers import anthropic

    short = (
        "<table><tr><th>Model</th><th>Base Input Tokens</th><th>Output Tokens</th></tr>"
        "<tr><td>Claude Opus 4.8</td><td>$5 / MTok</td><td>$25 / MTok</td></tr></table>"
    )
    rows = "".join(
        f"<tr><td>Model {i}</td><td>${i} / MTok</td><td>${i} / MTok</td></tr>" for i in range(8)
    )
    full = f"<table><tr><th>Model</th><th>Base Input Tokens</th><th>Output Tokens</th></tr>{rows}</table>"

    class FakeResp:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    pages = iter([short, full])

    class FakeClient:
        def get(self, *a, **k):
            return FakeResp(next(pages))

    monkeypatch.setattr(anthropic.time, "sleep", lambda *_: None)
    scraper = anthropic.AnthropicScraper(client=FakeClient())
    pricing = scraper._fetch_pricing()
    # first (short) page had 1 model < MIN; retried and kept the fuller parse
    assert len(pricing) >= anthropic.MIN_PRICED_MODELS


def test_arena_parse_overall_block_and_join():
    from scrape_providers import arena
    from scrape_providers.models import Model, Provider

    def row(rank, name, rating, votes):
        return (
            f'"rank":{rank},"rankUpper":{rank},"rankLower":{rank},'
            f'"modelKey":"{name}-text","modelDisplayName":"{name}",'
            f'"rating":{rating},"ratingUpper":{rating},"ratingLower":{rating},"votes":{votes}'
        )

    # overall block (ranks 1-3) then a second category restarting at rank 1
    html = (
        row(1, "claude-fable-5", 1507.59, 4297)
        + row(2, "gpt-5.5", 1474.85, 9000)
        + row(3, "claude-opus-4-8", 1477.85, 8000)
        + row(1, "gpt-5.5", 1490.0, 100)  # coding category: must be ignored
    ).replace('"', '\\"')

    scores = arena._parse(html)
    assert len(scores) == 3  # only the overall block
    assert scores["gpt-5.5"].elo == 1474.85  # not the coding-category 1490

    providers = [Provider(name="openai", models=[Model(id="gpt-5.5"), Model(id="gpt-5.5-pro")])]
    arena.annotate(providers, scores)
    models = {m.id: m for m in providers[0].models}
    assert models["gpt-5.5"].arena.rank == 2
    # exact-match only: gpt-5.5-pro must NOT inherit gpt-5.5's score
    assert models["gpt-5.5-pro"].arena is None


def test_deepseek_native_pricing_parse():
    from scrape_providers.providers.deepseek import _parse_pricing

    # transposed table: models as columns, with rowspan (PRICING) and colspan rows
    html = """
    <table>
      <tr><th>MODEL</th><td>deepseek-v4-flash(1)</td><td>deepseek-v4-pro</td></tr>
      <tr><td>CONTEXT LENGTH</td><td>1M</td></tr>
      <tr><td>MAX OUTPUT</td><td>MAXIMUM: 384K</td></tr>
      <tr><td>FEATURES</td><td>Json Output</td><td>✓</td><td>✓</td></tr>
      <tr><td>PRICING</td><td>1M INPUT TOKENS (CACHE HIT)</td><td>$0.0028</td><td>$0.003625</td></tr>
      <tr><td>1M INPUT TOKENS (CACHE MISS)</td><td>$0.14</td><td>$0.435</td></tr>
      <tr><td>1M OUTPUT TOKENS</td><td>$0.28</td><td>$0.87</td></tr>
    </table>
    """
    details = _parse_pricing(html)
    flash = details["deepseek-v4-flash"]  # footnote stripped
    assert flash["context_window"] == 1_000_000  # colspan broadcast
    assert flash["max_output_tokens"] == 384_000
    assert flash["pricing"].input == 0.14  # cache-miss is the headline input
    assert flash["pricing"].extra["cache_read"] == 0.0028  # cache-hit
    assert details["deepseek-v4-pro"]["pricing"].output == 0.87
    assert "json_output" in flash["capabilities"]


def test_catalog_validates_against_schema():
    import jsonschema

    from scrape_providers.emit import pruned_catalog
    from scrape_providers.models import ArenaScore, Endpoint, Model, Pricing, Provider
    from scrape_providers.schema import validate_catalog

    providers = [
        Provider(
            name="anthropic",
            root_url="https://api.anthropic.com",
            endpoints=[Endpoint(protocol="messages", endpoint="/v1/messages")],
            models=[
                Model(
                    id="claude-opus-4-8",
                    display_name="Claude Opus 4.8",
                    context_window=1_000_000,
                    max_output_tokens=128000,
                    modalities=["text", "image"],
                    open_source=False,
                    pricing=Pricing(input=5.0, output=25.0, extra={"cache_read": 0.5}),
                    arena=ArenaScore(rank=12, elo=1477.85, votes=13316),
                )
            ],
        )
    ]
    catalog = pruned_catalog(providers)
    validate_catalog(catalog)  # valid (incl. the agents section): should not raise
    assert catalog["agents"]  # agents section is emitted and schema-valid

    # an unknown protocol must fail validation
    bad = pruned_catalog(providers)
    bad["providers"][0]["endpoints"][0]["protocol"] = "grpc"
    with pytest.raises(jsonschema.ValidationError):
        validate_catalog(bad)

    # a stray key in an agent entry must fail (additionalProperties: false)
    bad_agent = pruned_catalog(providers)
    bad_agent["agents"][0]["bogus"] = 1
    with pytest.raises(jsonschema.ValidationError):
        validate_catalog(bad_agent)


def test_cli_validate_file(tmp_path, capsys):
    from scrape_providers.emit import to_yaml
    from scrape_providers.models import Endpoint, Model, Pricing, Provider

    provider = Provider(
        name="anthropic",
        root_url="https://api.anthropic.com",
        endpoints=[Endpoint(protocol="messages", endpoint="/v1/messages")],
        models=[
            Model(
                id="claude-opus-4-8",
                pricing=Pricing(input=5.0, output=25.0),
            )
        ],
    )
    good = tmp_path / "catalog.yaml"
    good.write_text(to_yaml([provider]))
    assert main(["--validate-file", str(good)]) == 0
    assert "valid" in capsys.readouterr().out

    # tamper: unknown protocol
    bad = tmp_path / "bad.yaml"
    bad.write_text(to_yaml([provider]).replace("protocol: messages", "protocol: telepathy"))
    assert main(["--validate-file", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "protocol" in err

    # non-existent file is reported, not crashed
    assert main(["--validate-file", str(tmp_path / "nope.yaml")]) == 1


def test_provider_names_have_no_slash():
    # --show splits PROVIDER/MODEL on the first '/', which requires this invariant
    assert all("/" not in name for name in registry.available())


def test_cli_show_argument_parsing(capsys):
    # missing slash -> usage error
    assert main(["--show", "deepseek-v4-pro"]) == 1
    assert "PROVIDER/MODEL" in capsys.readouterr().err
    # unknown provider
    assert main(["--show", "nope/x"]) == 1
    assert "unknown provider" in capsys.readouterr().err


def test_index_tools_normalizes_shapes():
    from scrape_providers.agent_profiles import index_tools

    # Anthropic-style raw tools array
    anthropic = [{"name": "Bash", "input_schema": {"type": "object", "properties": {}}}]
    assert index_tools(anthropic)["Bash"]["type"] == "object"

    # OpenAI responses-style, wrapped in {"tools": [...]}
    openai = {"tools": [{"name": "shell", "parameters": {"type": "object"}}]}
    assert "shell" in index_tools(openai)

    # OpenAI function-wrapped
    fn = [{"type": "function", "function": {"name": "apply_patch", "parameters": {"x": 1}}}]
    assert index_tools(fn)["apply_patch"] == {"x": 1}

    # already indexed
    assert index_tools({"read": {"type": "object"}})["read"]["type"] == "object"

    # typed built-in tools (no name) are kept, keyed by `type`
    builtins = [
        {"type": "web_search", "external_web_access": False},
        {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
    ]
    idx = index_tools(builtins)
    assert idx["web_search"]["external_web_access"] is False
    assert "exec_command" in idx


def test_cli_agent_tool_schema_no_capture(capsys, monkeypatch):
    from scrape_providers import agent_profiles

    # no vendored capture -> helpful error, exit 1 (not a crash)
    monkeypatch.setattr(agent_profiles, "load_schemas", lambda agent: {})
    assert main(["--agent-tool-schema", "codex"]) == 1
    assert "no vendored schemas" in capsys.readouterr().err
    # unknown agent -> error
    assert main(["--agent-tool-schema", "nope/Bash"]) == 1
    assert "unknown agent" in capsys.readouterr().err


def test_cli_agent_tool_schema_reads_capture(capsys):
    # the committed codex capture is read back, including built-in tools
    assert main(["--agent-tool-schema", "codex/web_search"]) == 0
    assert "web_search" in capsys.readouterr().out


def test_tool_names_prefers_capture(monkeypatch):
    from scrape_providers import agent_profiles

    # capture present -> names derived from it (sorted)
    monkeypatch.setattr(
        agent_profiles, "load_schemas", lambda a: {"exec_command": {}, "web_search": {}}
    )
    assert agent_profiles.tool_names("codex") == ["exec_command", "web_search"]
    assert agent_profiles.has_capture("codex")

    # no capture -> curated fallback
    monkeypatch.setattr(agent_profiles, "load_schemas", lambda a: {})
    assert agent_profiles.tool_names("codex") == agent_profiles.AGENT_PROFILES["codex"]["tools"]
    assert not agent_profiles.has_capture("codex")


def test_tool_descriptions_from_capture():
    from scrape_providers import agent_profiles

    descs = agent_profiles.tool_descriptions("codex")
    assert "apply_patch" in descs and descs["apply_patch"]  # has text
    assert descs.get("web_search", "") == ""  # built-in: no description


def test_extract_system_prompt_shapes():
    from scripts.capture.capture_tools import extract_system_prompt

    # Anthropic: system as text blocks
    assert (
        extract_system_prompt(
            {"system": [{"type": "text", "text": "You are Claude."}], "messages": []}
        )
        == "You are Claude."
    )
    # OpenAI Responses: instructions string
    assert extract_system_prompt({"instructions": "Be helpful.", "input": []}) == "Be helpful."
    # Chat completions: leading system turn, stops at first user turn
    assert (
        extract_system_prompt(
            {
                "messages": [
                    {"role": "system", "content": "Rules."},
                    {"role": "user", "content": "ignore me"},
                ]
            }
        )
        == "Rules."
    )


def test_cli_agent_system_prompt(capsys, monkeypatch, tmp_path):
    from scrape_providers import agent_profiles

    # no vendored prompt -> helpful error, exit 1
    monkeypatch.setattr(agent_profiles, "system_prompt", lambda agent: None)
    assert main(["--agent-system-prompt", "codex"]) == 1
    assert "no vendored system prompt" in capsys.readouterr().err
    # unknown agent -> error
    assert main(["--agent-system-prompt", "nope"]) == 1
    assert "unknown agent" in capsys.readouterr().err
    # present -> printed
    monkeypatch.setattr(agent_profiles, "system_prompt", lambda agent: "You are Codex.")
    assert main(["--agent-system-prompt", "codex"]) == 0
    assert "You are Codex." in capsys.readouterr().out


def test_cli_list_agents(capsys):
    assert main(["--list-agents"]) == 0
    out = capsys.readouterr().out
    assert "codex" in out and "claude_code" in out


def test_cli_list_agent_tools(capsys):
    # specific harness -> bare tool names, one per line (from the capture)
    assert main(["--list-agent-tools", "codex"]) == 0
    out = capsys.readouterr().out
    assert "apply_patch" in out and "exec_command" in out

    # no arg -> all harnesses with headers
    assert main(["--list-agent-tools"]) == 0
    out = capsys.readouterr().out
    assert "codex" in out and "claude_code" in out and "Bash" in out

    # unknown harness -> error, exit 1
    assert main(["--list-agent-tools", "nope"]) == 1
    assert "unknown agent" in capsys.readouterr().err


def test_cli_list_provider_models_unknown(capsys):
    assert main(["--list-provider-models", "nope"]) == 1
    assert "unknown provider" in capsys.readouterr().err


def test_print_and_set_curated_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SCRAPE_PROVIDERS_CURATED", str(tmp_path / "curated.yaml"))

    # no config yet -> prints the built-in defaults
    assert main(["--print-curated"]) == 0
    assert "claude-opus-4-8" in capsys.readouterr().out

    # set a custom list, then print should reflect it
    custom = tmp_path / "my.yaml"
    custom.write_text("anthropic:\n- claude-opus-4-8\n")
    assert main(["--set-curated", str(custom)]) == 0
    assert "saved to" in capsys.readouterr().out

    assert main(["--print-curated"]) == 0
    out = capsys.readouterr().out
    assert "claude-opus-4-8" in out
    assert "gpt-5.5" not in out  # replaced the defaults


def test_set_curated_rejects_bad_shape(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SCRAPE_PROVIDERS_CURATED", str(tmp_path / "curated.yaml"))
    bad = tmp_path / "bad.yaml"
    bad.write_text("anthropic: not-a-list\n")
    assert main(["--set-curated", str(bad)]) == 1
    assert "mapping of provider" in capsys.readouterr().err


def test_cli_schema_flag(capsys):
    assert main(["--schema"]) == 0
    out = capsys.readouterr().out
    assert '"models"' in out and '"providers"' in out


def test_cli_list_providers(capsys):
    assert main(["--list-providers"]) == 0
    assert "anthropic" in capsys.readouterr().out


def test_norm_strips_annotations():
    assert _norm("Claude Opus 4.8") == "claude opus 4.8"
    assert _norm("Claude Mythos 5 (limited availability)") == "claude mythos 5"


def test_parse_pricing_table():
    html = """
    <table>
      <tr><th>Model</th><th>Base Input Tokens</th>
          <th>Cache Hits &amp; Refreshes</th><th>Output Tokens</th></tr>
      <tr><td>Claude Opus 4.8</td><td>$5 / MTok</td>
          <td>$0.50 / MTok</td><td>$25 / MTok</td></tr>
    </table>
    """
    pricing = _parse_pricing_table(html)
    opus = pricing[_norm("Claude Opus 4.8")]
    assert opus.input == 5.0
    assert opus.output == 25.0
    assert opus.extra["cache_read"] == 0.5


# Live end-to-end test; only runs when an API key is configured.
def test_anthropic_live_scrape():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        import pytest

        pytest.skip("ANTHROPIC_API_KEY not set")
    with registry.get("anthropic")() as scraper:
        provider = scraper.scrape()
    assert provider.models
    assert all(m.id for m in provider.models)
    assert [e.protocol for e in provider.endpoints] == ["messages"]

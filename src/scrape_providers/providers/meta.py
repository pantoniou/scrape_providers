"""Meta Model API provider scraper — fully native (no OpenRouter).

Meta's Model API (served from ``api.meta.ai``, documented at ``dev.meta.ai``) is
a separate, billed product from Meta's open-weight Llama releases: an
OpenAI/Anthropic-SDK-compatible hosted API serving Meta's proprietary **Muse
Spark** family (``muse-spark-1.1``/``1.2``/``1.3``, plus discounted
``-contributor`` variants that trade lower pricing for training permission on
your traffic). Two native sources:

* ``GET https://api.meta.ai/v1/models`` (requires ``META_API_KEY``) — the ids
  actually served. Per the docs, this response carries no context window or
  modality info ("For a catalog of ... capabilities and context windows, see
  the Models page"), so those come from the docs instead.
* The docs' Models page (``dev.meta.ai/docs/models``) — a table of context
  window and input modalities per model id — and Pricing page
  (``dev.meta.ai/docs/pricing-rate-limits``) — per-tier (standard/contributor)
  token pricing. Both are fetched as their raw Markdown source (``<url>.md``)
  rather than rendered HTML: the docs site is client-rendered, and the ``.md``
  source is the same content in a much simpler, table-stable format to parse.

The pricing page prices a *tier*, not a model: each tier's section states its
member model ids in a "Models: `id`, `id`." sentence ahead of one shared price
table, rather than repeating a price per row, so the ids are read out of that
sentence and joined to the following table.

Muse Spark and Muse Wonder share a capability set, fixed across every version rather than
per-model-flagged (the docs describe them in prose, not a table): tool calling,
JSON-schema structured output, always-on reasoning with a `reasoning_effort`
knob, and web search grounding via the Responses API's `web_search` tool.

Meta also ships Muse Image (image generation) and Muse Voice Transcribe
(speech-to-text) on the same API, but neither fits this catalog's
per-million-token chat-model shape (image is priced per image, transcription
per audio-hour) — only the text models (Muse Spark and Muse Wonder) are scraped here.
"""

from __future__ import annotations

import os
import re

import httpx

from ..base import Scraper
from ..models import Model, Pricing, Provider
from ..tools import endpoint_for

API_ROOT = "https://api.meta.ai/v1"
MODELS_URL = f"{API_ROOT}/models"
DOCS_MODELS_URL = "https://dev.meta.ai/docs/models.md"
DOCS_PRICING_URL = "https://dev.meta.ai/docs/pricing-rate-limits.md"

# See "Muse Spark's capabilities are fixed..." above; emitted in the catalog's
# declared capability order via emit.normalize_capabilities.
_MUSE_SPARK_CAPABILITIES = [
    "tool_calling",
    "structured_outputs",
    "reasoning",
    "reasoning_effort",
    "web_search",
]

# The two text-model lines; both share the capability set above.
_TEXT_MODEL_PREFIXES = ("muse-spark-", "muse-wonder-")

_MODEL_ID = re.compile(r"`([a-z0-9][a-z0-9.\-]*)`")
_CONTEXT_TOKENS = re.compile(r"([\d,]+)-?\s*tokens?\s+context", re.IGNORECASE)
_INPUT_SENTENCE = re.compile(r"\btakes?\s+(.+?)\s+input\b", re.IGNORECASE)
_MONEY = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")
_H2 = re.compile(r"^##\s+.*$", re.MULTILINE)
_H3 = re.compile(r"^###\s+.*$", re.MULTILINE)
_MODELS_LINE = re.compile(r"^Models?:\s*(.+?)\s*$", re.MULTILINE)

# Pricing table row label -> where it lands in Pricing.
_FIELDS = {"input": "input", "output": "output"}
_EXTRA = {"cached input": "cache_read"}


class MetaScraper(Scraper):
    name = "meta"

    def scrape(self) -> Provider:
        docs = self._fetch_docs()  # model id -> {modalities, context_window, pricing}
        models = [
            Model(
                id=model_id,
                modalities=docs.get(model_id, {}).get("modalities") or ["text"],
                context_window=docs.get(model_id, {}).get("context_window"),
                capabilities=list(_MUSE_SPARK_CAPABILITIES),
                pricing=docs.get(model_id, {}).get("pricing"),
            )
            for model_id in self._fetch_model_ids()
            if model_id.startswith(_TEXT_MODEL_PREFIXES)
        ]
        return Provider(
            name=self.name,
            root_url=API_ROOT,
            endpoints=[
                endpoint_for(self.name, "chat_completions", "/chat/completions"),
                endpoint_for(self.name, "responses", "/responses"),
                endpoint_for(self.name, "messages", "/messages"),
            ],
            models=models,
        )

    def _fetch_model_ids(self) -> list[str]:
        key = os.environ.get("META_API_KEY")
        if not key:
            raise RuntimeError("META_API_KEY is required to scrape the Meta Model API")
        resp = self._client.get(
            MODELS_URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", []) if m.get("id")]

    def _fetch_docs(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for url, parse in (
            (DOCS_MODELS_URL, _parse_models_table),
            (DOCS_PRICING_URL, _parse_pricing),
        ):
            try:
                resp = self._client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
            except httpx.HTTPError:
                continue
            for model_id, info in parse(resp.text).items():
                out.setdefault(model_id, {}).update(info)
        return out


def _parse_models_table(markdown: str) -> dict[str, dict]:
    """Read context window + input modalities off each model line's prose section.

    The Models page has one ``## Muse X`` section per line; the sentence "takes
    text, image, ... input ... has a N-token context window" (or "All take ...")
    covers every version named in backticks inside that section.
    """
    out: dict[str, dict] = {}
    for body in _sections(markdown, _H2):
        ids = [i for i in _MODEL_ID.findall(body) if i.startswith(_TEXT_MODEL_PREFIXES)]
        inputs = _INPUT_SENTENCE.search(body)
        context = _CONTEXT_TOKENS.search(body)
        if not ids or inputs is None:
            continue
        modalities = [w for w in re.findall(r"[a-z]+", inputs.group(1).lower()) if w != "and"]
        info = {
            "modalities": modalities,
            "context_window": int(context.group(1).replace(",", "")) if context else None,
        }
        for model_id in ids:
            out[model_id] = dict(info)
    return out


def _parse_pricing(markdown: str) -> dict[str, dict]:
    """Join each pricing section's model list to its Standard/Contributor columns.

    A section states "Model(s): `a`, `b` in Standard mode; `c` in Contributor
    mode." ahead of one table with a column per mode, so each id takes the
    column of the mode its clause names (a ``-contributor`` id, failing that).
    """
    out: dict[str, dict] = {}
    for body in _sections(markdown, _H3):
        line = _MODELS_LINE.search(body)
        if line is None:
            continue
        columns = _tier_pricing(body)
        if not columns:
            continue
        for clause in line.group(1).split(";"):
            contributor = "contributor" in clause.lower()
            for model_id in _MODEL_ID.findall(clause):
                column = 1 if contributor or model_id.endswith("-contributor") else 0
                if column < len(columns) and columns[column] is not None:
                    out[model_id] = {"pricing": columns[column]}
    return out


def _sections(markdown: str, heading: re.Pattern[str]) -> list[str]:
    """Bodies of the sections opened by ``heading``, each up to the next heading of that level or higher."""
    starts = list(heading.finditer(markdown))
    level = len(markdown[starts[0].start() :].split(None, 1)[0]) if starts else 0
    stop = re.compile(rf"^#{{1,{level}}}\s", re.MULTILINE)
    bodies = []
    for m in starts:
        nxt = stop.search(markdown, m.end())
        bodies.append(markdown[m.end() : nxt.start() if nxt else len(markdown)])
    return bodies


def _tier_pricing(body: str) -> list[Pricing | None]:
    """One Pricing per value column of the section's usage table (None where unavailable)."""
    rows = _md_rows(body, ("usage",))
    width = max((len(r) for r in rows), default=1) - 1
    out: list[Pricing | None] = []
    for col in range(1, width + 1):
        fields: dict[str, float] = {}
        extra: dict[str, float] = {}
        for row in rows:
            if col >= len(row):
                continue
            label = row[0].strip().lower()
            value = _money(row[col])
            if value is None:
                continue
            if label in _FIELDS:
                fields[_FIELDS[label]] = value
            elif label in _EXTRA:
                extra[_EXTRA[label]] = value
        out.append(Pricing(**fields, extra=extra) if fields else None)
    return out


def _md_rows(markdown: str, header_prefix: tuple[str, ...]) -> list[list[str]]:
    """Body rows of the first Markdown pipe-table whose header cells start with ``header_prefix``."""
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        cells = _cells(line)
        if len(cells) >= len(header_prefix) and all(
            cells[j].lower().startswith(header_prefix[j]) for j in range(len(header_prefix))
        ):
            rows = []
            for row_line in lines[i + 2 :]:  # skip the header row and its "| :--- |" separator
                row_cells = _cells(row_line)
                if not row_cells:
                    break
                rows.append(row_cells)
            return rows
    return []


def _cells(line: str) -> list[str]:
    line = line.strip()
    if not line.startswith("|"):
        return []
    return [c.strip() for c in line.strip("|").split("|")]


def _money(text: str) -> float | None:
    m = _MONEY.search(text or "")
    return float(m.group(1)) if m else None

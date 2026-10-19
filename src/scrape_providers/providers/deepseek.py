"""DeepSeek provider scraper — fully native (no OpenRouter).

Two native sources:

* ``GET https://api.deepseek.com/models`` (requires ``DEEPSEEK_API_KEY``) — the
  authoritative list of model ids DeepSeek actually serves.
* The docs pricing page (``api-docs.deepseek.com``) — a transposed table (models
  as columns) giving cache-hit / cache-miss input pricing, output pricing,
  context length, max output, the model version, and the feature rows.

DeepSeek caching is automatic, so cache-hit (``cache_read``) pricing is published
for all their models; this is why we scrape DeepSeek directly rather than relying
on OpenRouter's spotty per-host cache figures.

Prices are tiered: each figure is split over an OFF-PEAK and a PEAK row, off-peak
being half the peak rate. The catalog takes the **peak (standard)** rate — what
you pay unless you deliberately schedule around the peak window (01:00-04:00 and
06:00-10:00 UTC on Chinese working days) — matching how the Google scraper keeps
Standard and drops Batch/Flex/Priority.

The English page was missing upstream for a while (checked 2026-09-07: their
server answers a missing path with the SPA index, so the fetch was a useless HTTP
200 and ``_parse_pricing`` found no ``MODEL`` table; ``build_catalog``'s
unpriced-models warning was the signal). It is back, with the tier rows above.
The ``zh-cn`` locale is used *only* for the model-version row (see
``_fetch_versions``) — it is priced in CNY, which would make the catalog's USD
figures FX-derived.

Web search and web fetch are two different things here, and only one of them is a
model capability:

* Model surfaces. The Responses API accepts the built-in ``web_search`` tool (the
  provider searches, adds the results to the context and continues the
  response), and only ``deepseek-v4-pro`` takes it — ``deepseek-flash`` ignores
  it. The Anthropic-format ``messages`` surface accepts Anthropic's server-side
  web search (``server_tool_use`` and ``web_search_tool_result`` are both
  supported). Those live on the endpoints in ``tools.py``, and ``web_search`` is
  a capability of the model that honours it.
* The standalone Search and Fetch APIs (``POST /v1/tools/search``,
  ``POST /v1/tools/fetch``): plain JSON, independent of any model and billed per
  successful request ($0.75 / 1,000) rather than per token. They are what the
  agent-harness integrations call server-side, so they are emitted as their own
  endpoints — protocols ``search`` and ``fetch`` — instead of being mistaken for
  a built-in chat tool.
"""

from __future__ import annotations

from datetime import date

import os
import re

import httpx
from selectolax.parser import HTMLParser

from ..base import Scraper
from ..models import Model, Pricing, Provider
from ..tools import endpoint_for

MODELS_URL = "https://api.deepseek.com/models"
# The zh-cn locale of the same doc. Used *only* for the model-version row (see
# `_fetch_versions`), never for prices: that table is in CNY and peak/off-peak
# tiered, whereas the version cell is a plain ASCII build name in either locale.
VERSIONS_URL = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing"
PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing"

# "模型版本" — the only row on the page that is a version marker.
_VERSION_ROW = "版本"
# The build suffix on a version name: DeepSeek-V4-Flash-0731 -> July 31.
_BUILD_DATE = re.compile(r"-(\d{2})(\d{2})$")
_MONEY = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")
_FOOTNOTE = re.compile(r"\(.*?\)|\s+")  # strip "(1)" footnotes / whitespace from ids
# Price tiers, as their own rows under a price row (see `_parse_pricing`). The
# peak rate is the standard one; off-peak is a discount for scheduling around it.
_TIERS = ("PEAK", "OFF-PEAK")
_STANDARD_TIER = "PEAK"
# The Responses API's built-in `web_search` tool is taken by one model only:
# "Ignored with deepseek-flash" (see the Models & Pricing feature table, which
# lists no search row, and the Responses API guide's Tools table).
_WEB_SEARCH_MODELS = frozenset({"deepseek-v4-pro"})


class DeepSeekScraper(Scraper):
    name = "deepseek"

    def scrape(self) -> Provider:
        details = self._fetch_pricing()  # id -> parsed characteristics + pricing
        versions = self._fetch_versions()  # id -> in-place revision date
        models = _build_models(self._fetch_model_ids(), details, versions)
        return Provider(
            name=self.name,
            root_url="https://api.deepseek.com",
            endpoints=[
                endpoint_for(self.name, "chat_completions", "/chat/completions"),
                # The Responses API, in the OpenAI Responses format (stateless).
                endpoint_for(self.name, "responses", "/responses"),
                # DeepSeek also exposes an Anthropic-format (Messages API) surface.
                endpoint_for(self.name, "messages", "/anthropic/v1/messages"),
                # The standalone Search & Fetch APIs: raw JSON, no model in the
                # request. Priced per request rather than per token, so they are
                # endpoints of their own (see tools.py).
                endpoint_for(self.name, "search", "/v1/tools/search"),
                endpoint_for(self.name, "fetch", "/v1/tools/fetch"),
            ],
            models=models,
        )

    def _fetch_model_ids(self) -> list[str]:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is required to scrape the DeepSeek models API")
        resp = self._client.get(
            MODELS_URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def _fetch_versions(self) -> dict[str, date]:
        """Best-effort model-version row from the zh-cn docs (see VERSIONS_URL)."""
        try:
            resp = self._client.get(VERSIONS_URL, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        return parse_versions(resp.text)

    def _fetch_pricing(self) -> dict[str, dict]:
        try:
            resp = self._client.get(PRICING_URL, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        return _parse_pricing(resp.text)


def _build_models(
    model_ids: list[str], details: dict[str, dict], versions: dict[str, date]
) -> list[Model]:
    """Join the served ids to the pricing page's view of them."""
    models: list[Model] = []
    for model_id in model_ids:
        d = details.get(model_id, {})
        capabilities = list(d.get("capabilities", []))
        if model_id in _WEB_SEARCH_MODELS:
            # Its endpoint carries the tool (tools.py); naming it here makes it
            # a capability of the model, which is what the merged catalog entry
            # publishes.
            capabilities.append("web_search")
        models.append(
            Model(
                id=model_id,
                display_name=d.get("display_name"),
                context_window=d.get("context_window"),
                max_output_tokens=d.get("max_output_tokens"),
                modalities=["text"],
                capabilities=capabilities,
                open_source=True,
                updated=versions.get(model_id),
                pricing=d.get("pricing"),
            )
        )
    return models


def parse_versions(html: str, today: date | None = None) -> dict[str, date]:
    """``model id -> in-place revision date`` from the docs' model-version row.

    DeepSeek serves stable ids whose weights are swapped underneath them — the
    docs state that ``deepseek-v4-flash`` currently runs ``DeepSeek-V4-Flash-0731``
    — which is exactly the in-place revision the catalog's ``updated`` field
    means. The row sits in the same transposed table as everything else, so the
    model ids come from the header and the versions from the row beneath.

    The build suffix carries no year (``-0731``), so it is resolved to the most
    recent such month/day that is not in the future. A revision the docs describe
    as current cannot postdate today, and DeepSeek has never kept a build listed
    for over a year, so this is unambiguous in practice.
    """
    tree = HTMLParser(html)
    for table in tree.css("table"):
        rows = [[c.text(strip=True) for c in tr.css("th,td")] for tr in table.css("tr")]
        if not rows or not rows[0]:
            continue
        ids = [_clean_id(c) for c in rows[0][1:]]
        for row in rows[1:]:
            if not row or _VERSION_ROW not in row[0]:
                continue
            today = today or date.today()
            versions: dict[str, date] = {}
            for model_id, cell in zip(ids, row[1:]):
                match = _BUILD_DATE.search(cell)
                if not match:
                    continue  # e.g. an experimental build with no dated suffix
                month, day = int(match.group(1)), int(match.group(2))
                for year in (today.year, today.year - 1):
                    try:
                        when = date(year, month, day)
                    except ValueError:
                        break  # not a real month/day; not a build date after all
                    if when <= today:
                        versions[model_id] = when
                        break
            return versions
    return {}


def _parse_pricing(html: str) -> dict[str, dict]:
    """Parse the transposed pricing table into ``model id -> details`` dict.

    Each price is split over an ``OFF-PEAK`` and a ``PEAK`` row beneath the row
    that names it (``1M OUTPUT TOKENS`` and so on). Only the peak (standard) rate
    is taken; see the module docstring.
    """
    tree = HTMLParser(html)
    table = tree.css_first("table")
    if table is None:
        return {}
    rows = [[c.text(strip=True) for c in tr.css("th,td")] for tr in table.css("tr")]
    if not rows or not rows[0] or rows[0][0].upper() != "MODEL":
        return {}

    model_ids = [_clean_id(c) for c in rows[0][1:]]
    n = len(model_ids)
    out: dict[str, dict] = {mid: {"capabilities": []} for mid in model_ids}
    # Per-model price components, filled as the relevant rows are found.
    inputs: list[float | None] = [None] * n
    outputs: list[float | None] = [None] * n
    cache: list[float | None] = [None] * n

    section = ""  # the price row a tier sub-row prices
    for row in rows[1:]:
        label, values = _label_values(row, n)
        if values is None:
            continue
        up = label.upper().strip()
        tier = ""
        if up in _TIERS:  # a sub-row: PEAK / OFF-PEAK beneath the row it prices
            up, tier = section, up
        else:
            head, _, tail = up.rpartition(" ")
            if tail in _TIERS:  # a price row that names its own tier
                up, tier = head, tail
            section = up
        if tier and tier != _STANDARD_TIER:
            continue  # a discounted tier, not the rate you normally pay
        if "CACHE HIT" in up:
            cache = [_money(v) for v in values]
        elif "CACHE MISS" in up or (up.startswith("1M INPUT") and "CACHE" not in up):
            inputs = [_money(v) for v in values]
        elif "OUTPUT TOKEN" in up:
            outputs = [_money(v) for v in values]
        elif up == "CONTEXT LENGTH":
            for mid, v in zip(model_ids, _broadcast(values, n)):
                out[mid]["context_window"] = _tokens(v)
        elif up == "MAX OUTPUT":
            for mid, v in zip(model_ids, _broadcast(values, n)):
                out[mid]["max_output_tokens"] = _tokens(v)
        elif up == "MODEL VERSION":
            for mid, v in zip(model_ids, _broadcast(values, n)):
                out[mid]["display_name"] = v
        elif all(v == "✓" for v in values) and values:
            cap = _slug(label)
            for mid, v in zip(model_ids, _broadcast(values, n)):
                if v == "✓" and cap:
                    out[mid]["capabilities"].append(cap)

    for i, mid in enumerate(model_ids):
        out[mid]["capabilities"] = sorted(out[mid]["capabilities"])
        if inputs[i] is not None or outputs[i] is not None:
            extra = {"cache_read": cache[i]} if cache[i] is not None else {}
            out[mid]["pricing"] = Pricing(input=inputs[i], output=outputs[i], extra=extra)
    return out


def _label_values(row: list[str], n: int) -> tuple[str, list[str] | None]:
    """Split a table row into (label, n per-model value cells).

    Rows may carry a leading section header cell (rowspan) so the label can span
    multiple leading cells; a single trailing value indicates a colspan that
    applies to every model.
    """
    if len(row) >= n + 1:
        return " ".join(row[:-n]), row[-n:]
    if len(row) == 2:  # single colspan value
        return row[0], [row[1]]
    return "", None


def _broadcast(values: list[str], n: int) -> list[str]:
    return values if len(values) == n else values * n


def _clean_id(text: str) -> str:
    return _FOOTNOTE.sub("", text)


def _slug(label: str) -> str:
    label = re.sub(r"[(（].*?[)）]", "", label)  # drop ascii and full-width parentheticals
    label = re.sub(r"^\s*features\b", "", label, flags=re.IGNORECASE)  # drop section prefix
    return "_".join(label.lower().split())


def _money(text: str) -> float | None:
    m = _MONEY.search(text or "")
    return float(m.group(1)) if m else None


def _tokens(text: str) -> int | None:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([MK])?", text or "")
    if not m:
        return None
    value = float(m.group(1))
    return int(value * {"M": 1_000_000, "K": 1000}.get(m.group(2), 1))

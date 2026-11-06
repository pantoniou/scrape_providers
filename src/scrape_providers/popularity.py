"""OpenRouter popularity: which models people actually route traffic to.

``/api/v1/models`` carries no usage figures, but the public rankings page
(``openrouter.ai/rankings``) embeds, in its Next.js payload, the per-day token
volume of the most-used models. Like the LMArena scrape in ``arena.py`` this is
read out of escaped JSON in the HTML, so it can break without notice — callers
treat an empty result as "no popularity signal" rather than as an error.

A ranking row names a *permaslug* (``tencent/hy4-20261104``) and a variant
(``standard``/``free``), not an id. ``canonical_slug`` on the models API is the
same permaslug, but several ids share one (``openai/gpt-6-luna`` and its
``:batch`` twin), so the row's variant picks which: ``free`` selects the
``:free`` id, anything else the unsuffixed one.
"""

from __future__ import annotations

from collections import defaultdict

import httpx
import re

RANKINGS_URL = "https://openrouter.ai/rankings"
MODELS_URL = "https://openrouter.ai/api/v1/models"

_ROW = re.compile(
    r'\{"date":"[^"]+","model_permaslug":"([^"]+)","variant":"([^"]+)",'
    r'"total_completion_tokens":(\d+),"total_prompt_tokens":(\d+)'
)


def fetch_ranking(client: httpx.Client) -> list[str]:
    """OpenRouter model ids, most-used first; ``[]`` if the page can't be read."""
    try:
        page = client.get(RANKINGS_URL, headers={"User-Agent": "Mozilla/5.0"})
        page.raise_for_status()
        api = client.get(MODELS_URL)
        api.raise_for_status()
        ids = [m["id"] for m in api.json().get("data", []) if m.get("id")]
        slugs = {m["id"]: m.get("canonical_slug") for m in api.json().get("data", [])}
    except (httpx.HTTPError, ValueError):
        return []
    return resolve(parse_rankings(page.text), ids, slugs)


def parse_rankings(html: str) -> list[tuple[str, str, int]]:
    """``(permaslug, variant, tokens)`` summed over every day in the payload."""
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for slug, variant, completion, prompt in _ROW.findall(html.replace('\\"', '"')):
        totals[(slug, variant)] += int(completion) + int(prompt)
    return [(s, v, t) for (s, v), t in sorted(totals.items(), key=lambda kv: -kv[1])]


def resolve(
    rows: list[tuple[str, str, int]], ids: list[str], slugs: dict[str, str | None]
) -> list[str]:
    """Map ranking rows onto served ids, most-used first, one entry per id."""
    by_slug: dict[str, list[str]] = defaultdict(list)
    for model_id in ids:
        by_slug[slugs.get(model_id) or model_id].append(model_id)
    ranked: list[str] = []
    for slug, variant, _tokens in rows:
        candidates = by_slug.get(slug) or by_slug.get(slug.rsplit("-", 1)[0], [])
        if not candidates:
            continue
        suffix = ":free" if variant == "free" else None
        pick = [c for c in candidates if (c.endswith(suffix) if suffix else ":" not in c)]
        chosen = min(pick or candidates, key=len)
        if chosen not in ranked:
            ranked.append(chosen)
    return ranked

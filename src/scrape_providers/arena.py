"""LMArena (Chatbot Arena) Elo / rank scraping and joining.

LMArena's leaderboard page (``lmarena.ai/leaderboard``) is a Next.js app that
embeds its data as escaped JSON in the streamed RSC payload — there is no public
JSON API (the ``/api`` routes reject non-browser requests). We extract the
``overall`` leaderboard rows from the page markup.

The page lists several category leaderboards (overall, coding, vision, …) back
to back, each starting at rank 1; the ``overall`` block is the first contiguous
run of increasing ranks, which is what :func:`fetch_scores` returns.

Scores are joined onto catalog models by **exact** (normalized) model name, with
one narrow exception: the board scores reasoning-effort *variants* of a model as
their own rows (``claude-opus-4-8-high``, ``gpt-5.6-sol-xhigh``,
``gemini-3-flash (thinking-minimal)``). Those are the same model run at a
different effort, not a different model, so :data:`_EFFORT_SUFFIX` — a closed
vocabulary of effort words, never a general pattern — strips them and the row
falls back onto the base model. An exact row always wins over a stripped one, so
a variant only ever fills a model the board doesn't otherwise score; where
several variants collapse onto one unscored model, the strongest is kept.

Fuzzy matching beyond that is deliberately avoided — a wrong Elo is worse than a
missing one (e.g. ``gpt-5.5-pro`` must not inherit ``gpt-5.5``'s rating). The
suffix list is enumerated precisely so it cannot drift into that: ``-pro`` names
a different model, ``-high`` names a knob.
"""

from __future__ import annotations

import re
import time

import httpx

from .models import ArenaScore, Model, Provider

ARENA_URL = "https://lmarena.ai/leaderboard"
RETRIES = 4
RETRY_DELAY = 0.5
MIN_MODELS = 20  # the overall board has ~200; a short parse means a partial page

_ROW = re.compile(
    r'"rank":(\d+),"rankUpper":\d+,"rankLower":\d+,'
    r'"modelKey":"[^"]*","modelDisplayName":"([^"]+)",'
    r'"rating":([\d.]+),"ratingUpper":[\d.]+,"ratingLower":[\d.]+,"votes":(\d+)'
)
_DATE_SUFFIX = re.compile(r"-\d{6,8}$")
# Reasoning-effort variants the board scores as separate rows. Enumerated, not a
# general suffix pattern: every word here is an effort setting of the same model.
_EFFORT = r"x?high|max|medium|low|minimal|thinking(?:-[a-z]+)*"
_EFFORT_SUFFIX = re.compile(
    # `-high`, `-thinking`, optionally with a context qualifier (`-high-32k`),
    # or the parenthesized form (`(xhigh)`, `(thinking-minimal)`).
    rf"(?:[\s-]*\((?:{_EFFORT})\)|-(?:{_EFFORT})(?:-\d+k)?)$"
)


def fetch_scores(client: httpx.Client) -> dict[str, ArenaScore]:
    """Return ``normalized model name -> ArenaScore`` for the overall board."""
    best: dict[str, ArenaScore] = {}
    for attempt in range(RETRIES):
        try:
            resp = client.get(ARENA_URL, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            scores = _parse(resp.text)
        except httpx.HTTPError:
            scores = {}
        if len(scores) > len(best):
            best = scores
        if len(best) >= MIN_MODELS:
            break
        if attempt < RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    return best


def _parse(html: str) -> dict[str, ArenaScore]:
    text = html.replace('\\"', '"')
    scores: dict[str, ArenaScore] = {}
    prev_rank = 0
    for rank_s, name, rating_s, votes_s in _ROW.findall(text):
        rank = int(rank_s)
        if rank <= prev_rank:
            break  # rank reset -> end of the overall block, next category begins
        prev_rank = rank
        scores[_norm(name)] = ArenaScore(rank=rank, elo=float(rating_s), votes=int(votes_s))
    return scores


def annotate(providers: list[Provider], scores: dict[str, ArenaScore]) -> None:
    """Attach arena scores to models in place, matching on exact normalized name."""
    variants = _effort_variants(scores)
    for provider in providers:
        for model in provider.models:
            score = _lookup(model, scores, variants)
            if score is not None:
                model.arena = score


def _strip_effort(key: str) -> str:
    """Drop trailing effort suffixes (``claude-opus-5-high`` -> ``claude-opus-5``)."""
    prev = None
    while prev != key:
        prev = key
        key = _EFFORT_SUFFIX.sub("", key).strip()
        key = _DATE_SUFFIX.sub("", key) if key != prev else key
    return key


def _effort_variants(scores: dict[str, ArenaScore]) -> dict[str, ArenaScore]:
    """Base name -> best score among the board's effort variants of it.

    Rows that are already a plain model name contribute nothing; only rows the
    exact join can't use are folded back onto their base. Where a model has
    several variants (``-high`` and ``-max``), the strongest is kept — they are
    the same model, so the spread is a knob setting, not a different contender.
    """
    best: dict[str, ArenaScore] = {}
    for name, score in scores.items():
        base = _strip_effort(name)
        if base == name or not base:
            continue  # a plain model name; the exact join already handles it
        # The two suffixes appear in either order (`…-high-32k`,
        # `deepseek-v4-pro-high-20260813`), so index the dated base as well —
        # catalog ids reach `_lookup` both with and without their date.
        for key in {base, _DATE_SUFFIX.sub("", base)}:
            current = best.get(key)
            if current is None or score.elo > current.elo:
                best[key] = score
    return best


def _lookup(
    model: Model,
    scores: dict[str, ArenaScore],
    variants: dict[str, ArenaScore] | None = None,
) -> ArenaScore | None:
    key = _norm(model.id)
    if key in scores:
        return scores[key]
    stripped = _DATE_SUFFIX.sub("", key)
    if stripped in scores:
        return scores[stripped]
    # Only now fall back to an effort variant: an exact row always wins.
    if variants:
        return variants.get(key) or variants.get(stripped)
    return None


def _norm(name: str) -> str:
    """Lowercase and drop any vendor prefix (``z-ai/glm-5.2`` -> ``glm-5.2``)."""
    return name.split("/")[-1].strip().lower()

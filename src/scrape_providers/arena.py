"""LMArena (Chatbot Arena) Elo / rank scraping and joining.

LMArena's leaderboard page (``lmarena.ai/leaderboard``) is a Next.js app that
embeds its data as escaped JSON in the streamed RSC payload — there is no public
JSON API (the ``/api`` routes reject non-browser requests). We extract the
``overall`` leaderboard rows from the page markup.

The page lists several category leaderboards (overall, coding, vision, …) back
to back, each starting at rank 1; the ``overall`` block is the first contiguous
run of increasing ranks, which is what :func:`fetch_scores` returns.

Scores are joined onto catalog models by **exact** (normalized) model name only.
Fuzzy matching is deliberately avoided — a wrong Elo is worse than a missing one
(e.g. ``gpt-5.5-pro`` must not inherit ``gpt-5.5``'s rating).
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
    for provider in providers:
        for model in provider.models:
            score = _lookup(model, scores)
            if score is not None:
                model.arena = score


def _lookup(model: Model, scores: dict[str, ArenaScore]) -> ArenaScore | None:
    key = _norm(model.id)
    if key in scores:
        return scores[key]
    stripped = _DATE_SUFFIX.sub("", key)
    return scores.get(stripped)


def _norm(name: str) -> str:
    """Lowercase and drop any vendor prefix (``z-ai/glm-5.2`` -> ``glm-5.2``)."""
    return name.split("/")[-1].strip().lower()

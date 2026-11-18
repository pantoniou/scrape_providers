"""Rank current first-party model tiers and reseller offerings by persona."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

# Emission order for the personas map.
PERSONAS = ("orchestrate", "plan", "code", "explore", "review", "summarize")


@dataclass(frozen=True)
class Persona:
    """How one persona trades strength off against cost.

    ``strength_weight`` is its share of the final score (the rest goes to cost),
    so 0.9 means "take the strongest and barely look at the bill" and 0.15 means
    "take the cheapest thing that works". ``input_share`` is the fraction of the
    persona's tokens that are input, which sets how its blended price is built.
    """

    name: str
    strength_weight: float
    input_share: float


_PROFILES = (
    # Directs the work and sees every result: capability dominates, and its own
    # token volume is small next to the sub-agents it dispatches.
    Persona("orchestrate", strength_weight=0.90, input_share=0.5),
    # Reasons at length before acting, so it is output-heavy and worth paying for.
    Persona("plan", strength_weight=0.85, input_share=0.35),
    # Reads a fair amount of context and writes real diffs.
    Persona("code", strength_weight=0.70, input_share=0.6),
    # Searches and reads far more than it writes; runs often, so price matters.
    Persona("explore", strength_weight=0.30, input_share=0.85),
    # Reads a diff and writes a short verdict, but a weak reviewer is worthless.
    Persona("review", strength_weight=0.60, input_share=0.8),
    # The cheapest competent model: almost all input, a paragraph of output.
    Persona("summarize", strength_weight=0.15, input_share=0.9),
)


# Each provider's chat families, from least to most capable. Full matches
# keep aliases, paid effort variants, batch, and specialist models out.
_NATIVE_TIERS = {
    "anthropic": tuple(
        rf"claude-{family}-(?P<version>\d+(?:\.\d+)*)"
        for family in ("haiku", "sonnet", "opus", "fable")
    ),
    "openai": tuple(
        rf"gpt-(?P<version>\d+(?:\.\d+)*)-{family}"
        for family in ("mini", "luna", "terra", "sol", "astra")
    ),
    "google": tuple(
        rf"gemini-(?P<version>\d+(?:\.\d+)*)-{family}"
        for family in ("flash-lite", "(?:flash|hydrogen)", "argon", "(?:pro|carbon)")
    ),
    "meta": tuple(
        rf"muse-{family}-(?P<version>\d+(?:\.\d+)*)" for family in ("ember", "spark", "wonder")
    ),
    "deepseek": (
        r"deepseek(?:-v(?P<version>\d+(?:\.\d+)*))?-flash",
        r"deepseek(?:-v(?P<version>\d+(?:\.\d+)*))?-pro",
    ),
}
_MAX_NATIVE_CHOICES = 3


def _native_personas(
    offerings: list[dict], tiers: tuple[str, ...], introduced: dict[str, str]
) -> dict[str, list[str]]:
    current = []
    for pattern in tiers:
        candidates = []
        for offering in offerings:
            cid = offering["canonical_id"]
            match = re.fullmatch(pattern, cid)
            if not match:
                continue
            version = tuple(int(part) for part in (match["version"] or "0").split("."))
            model_id = offering["provider_model_id"]
            # Version ties favor stable dispatch IDs over dated snapshots.
            snapshot = re.search(r"-(\d{8})$", model_id)
            candidates.append((version, snapshot is None, -_recency(cid, introduced), model_id))
        if candidates:
            current.append(max(candidates)[-1])
    if not current:
        return {}

    result = {}
    for persona in _PROFILES:
        target = round(persona.strength_weight * (len(current) - 1))
        nearest = sorted(range(len(current)), key=lambda tier: (abs(tier - target), -tier))
        result[persona.name] = [current[tier] for tier in nearest[:_MAX_NATIVE_CHOICES]]
    return result


# Suffixes naming a *variant* of a model rather than a different model. Two
# kinds, both of which keep the base model's strength: packaging variants that
# change only the price or the terms (`-contributor` buys a discount with
# training rights, `:free`/`-free` a promotional zero, `:batch` a slower queue)
# and effort/recency variants that are the same weights turned up or aliased
# (`-pro`, `-max`, `-latest`). Enumerated, never a general suffix rule.
_VARIANT_SUFFIX = re.compile(
    r"[-:](?:contributor|latest|preview|pro|max|free|batch|exp|thinking|it|instruct)$"
)
# Parameter counts are part of the model's identity, not a version: gpt-oss-20b
# must never inherit gpt-oss-120b's rating. Protected before versions are dropped.
_SIZE = re.compile(r"\d+(?:\.\d+)?[bm](?=-|$)")
_VERSION = re.compile(r"-v?\d+(?:\.\d+)*(?=-|$)")


def _base_ids(cid: str) -> list[str]:
    """``cid`` and the successively less specific ids it is a variant of."""
    forms = [cid]
    stem = cid.split(":")[0]
    if stem != cid:
        forms.append(stem)
    while True:
        shorter = _VARIANT_SUFFIX.sub("", stem)
        if shorter == stem:
            break
        stem = shorter
        forms.append(stem)
    return forms


def _family(cid: str) -> str:
    """The version-independent family a canonical id belongs to.

    ``muse-spark-1.3`` and ``muse-spark-1.2`` share a family; ``claude-haiku-4.5``
    and ``claude-opus-5`` do not, and neither do ``gpt-oss-20b`` and
    ``gpt-oss-120b``.
    """
    stem = _base_ids(cid)[-1]
    kept = _SIZE.findall(stem)
    stem = _SIZE.sub("\0", stem)
    stem = _VERSION.sub("", stem)
    for size in kept:
        stem = stem.replace("\0", size, 1)
    return stem.strip("-")


def resolve_strength(canonical_id: str, strength: dict[str, float]) -> float | None:
    """The best-evidenced strength for a model the board may not score directly.

    Three steps, most direct first: the model's own Elo; the Elo of the model it
    is a variant of (``muse-spark-1.3-contributor`` is ``muse-spark-1.3`` at a
    discount, ``~claude-opus-latest`` is whichever Opus is current); failing
    both, the strongest scored member of its family, which for a *newer* version
    of a scored line is a floor rather than a guess — vendors do not ship a 1.3
    weaker than the 1.2 it replaces.

    Returns ``None`` when nothing in the family is scored, leaving the caller to
    fall back to price.
    """
    for form in _base_ids(canonical_id):
        if form in strength:
            return strength[form]
    family = _family(canonical_id)
    if not family:
        return None
    siblings = [elo for cid, elo in strength.items() if _family(cid) == family]
    return max(siblings) if siblings else None


def _blended_cost(pricing: dict | None, input_share: float) -> float | None:
    """The offering's price for one persona's token mix, or ``None`` if unpriced."""
    if not pricing:
        return None
    inp, out = pricing.get("input"), pricing.get("output")
    if inp is None or out is None:
        return None
    return input_share * inp + (1.0 - input_share) * out


def _scale(values: list[float]) -> "callable":
    """Map values onto 0..1 by their own range (all-equal -> a flat 0.5)."""
    low, high = min(values), max(values)
    if high == low:
        return lambda _v: 0.5
    return lambda v: (v - low) / (high - low)


def _recency(canonical_id: str, introduced: dict[str, str]) -> float:
    """Sort key placing the newer model first, and undated models last.

    Only ever a tie-break. Strength inherited from a family is a *floor* shared
    by every member, so a newer version and the older one it replaces can score
    identically at the same price — and between two models that look equal on
    both axes, the newer is the better default. Negated because the surrounding
    sorts ascend.
    """
    # A packaging variant is usually too marginal to get its own catalog entry
    # with a date, so fall back along the same chain strength uses: the release
    # date of `muse-spark-1.3` is the release date of its `-contributor` too.
    when = next((introduced[f] for f in _base_ids(canonical_id) if f in introduced), None)
    if not when:
        return 0.0  # larger than any negated ordinal, so undated sorts last
    try:
        return -date.fromisoformat(when).toordinal()
    except (TypeError, ValueError):
        return 0.0


def build_personas(
    offerings: list[dict],
    strength: dict[str, float],
    introduced: dict[str, str] | None = None,
    *,
    provider: str | None = None,
) -> dict[str, list[str]]:
    """Return dispatch IDs in preference order for each persona."""
    introduced = introduced or {}
    if provider in _NATIVE_TIERS:
        return _native_personas(offerings, _NATIVE_TIERS[provider], introduced)
    resolved = [
        (
            o["provider_model_id"],
            resolve_strength(o.get("canonical_id", ""), strength),
            o.get("pricing"),
            _recency(o.get("canonical_id", ""), introduced),
        )
        for o in offerings
    ]
    result: dict[str, list[str]] = {}
    for persona in _PROFILES:
        scored: list[tuple[str, float, float, float]] = []
        unscored: list[tuple[str, float, float]] = []
        for model_id, elo, pricing, recency in resolved:
            cost = _blended_cost(pricing, persona.input_share)
            if cost is None:
                continue
            if elo is None:
                unscored.append((model_id, cost, recency))
            else:
                scored.append((model_id, elo, cost, recency))
        if not scored and not unscored:
            continue
        weight = persona.strength_weight
        ranked: list[str] = []

        if scored:
            scale_elo = _scale([c[1] for c in scored])
            scale_cost = _scale([c[2] for c in scored])
            # Cost enters negated: cheaper is better, and a persona weighting
            # strength at 0.9 spends only 0.1 of its score caring about price.
            ranked += [
                c[0]
                for c in sorted(
                    scored,
                    key=lambda c: (
                        -(weight * scale_elo(c[1]) - (1.0 - weight) * scale_cost(c[2])),
                        c[2],
                        c[3],  # equal on both axes: prefer the newer model
                        c[0],
                    ),
                )
            ]
        if unscored:
            # Nothing in this model's family is scored, so price is the only
            # signal left. Used as a stand-in for both axes at once, it collapses
            # to: a strength-led persona (weight > 0.5) prefers the dearer model,
            # a cost-led one the cheaper — and the whole group sits below the
            # models whose strength is actually evidenced.
            scale_cost = _scale([c[1] for c in unscored])
            ranked += [
                c[0]
                for c in sorted(
                    unscored,
                    key=lambda c: (-((2.0 * weight - 1.0) * scale_cost(c[1])), c[2], c[0]),
                )
            ]
        result[persona.name] = ranked
    return result

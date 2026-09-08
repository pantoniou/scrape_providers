"""Per-provider persona rankings: which of a provider's models to reach for.

An orchestrator picking models inside one provider's lineup asks a narrower
question than "what is the best model" — it asks "which of *these* should run
this step". The answer differs per step (a planner wants the strongest model
almost regardless of price; a file-reading explorer wants the cheapest one that
is still competent) and per provider, since each serves a different lineup at
its own prices.

So each provider entry carries a ``personas`` map: persona -> that provider's
models in preference order. Two things drive the order.

**Strength** is LMArena Elo, taken from the model's merged catalog entry rather
than from the offering, so every provider serving a scored model inherits its
score (Fireworks' ``glm-5p3`` is the ``glm-5.3`` the board ranks). Models the
board doesn't score are left out rather than guessed at: the only other signal
available is price, and imputing strength from price would make the two axes the
same axis and the ranking circular.

**Cost** is a blend of the offering's own input and output prices, weighted per
persona, because personas have genuinely different token shapes — a reviewer
reads far more than it writes, a planner writes long reasoning traces. Prices are
the provider's, so the same model can rank differently across providers.

Models are then reduced to the **Pareto frontier**: one that is both weaker and
more expensive than another of the same provider's models is dominated and never
worth dispatching to, so it is dropped rather than ranked last. What remains is
ordered by the persona's own strength/cost trade-off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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


def build_personas(offerings: list[dict], strength: dict[str, float]) -> dict[str, list[str]]:
    """``persona -> provider model ids, best first`` for one provider's lineup.

    ``offerings`` are the provider's catalog offerings (``canonical_id``,
    ``provider_model_id``, ``pricing``); ``strength`` maps canonical id -> Elo.

    Every priced offering is ranked — a weaker or dearer model is ordered lower,
    never dropped, since "which of this lineup runs this step" wants the whole
    lineup tiered, and the last resort still has to be nameable. Offerings split
    into two tiers: those with a strength (measured, or inherited from the model
    they are a variant of) rank above those without, and the remainder fall back
    to price, which for a persona that values strength means dearest first and
    for a cost-led one means cheapest first. An offering with no usable price
    can't be placed on either axis and is the only kind left out.
    """
    resolved = [
        (
            o["provider_model_id"],
            resolve_strength(o.get("canonical_id", ""), strength),
            o.get("pricing"),
        )
        for o in offerings
    ]
    result: dict[str, list[str]] = {}
    for persona in _PROFILES:
        scored: list[tuple[str, float, float]] = []
        unscored: list[tuple[str, float]] = []
        for model_id, elo, pricing in resolved:
            cost = _blended_cost(pricing, persona.input_share)
            if cost is None:
                continue
            if elo is None:
                unscored.append((model_id, cost))
            else:
                scored.append((model_id, elo, cost))
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
                    unscored, key=lambda c: (-((2.0 * weight - 1.0) * scale_cost(c[1])), c[0])
                )
            ]
        result[persona.name] = ranked
    return result

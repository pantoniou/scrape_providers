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


def _frontier(candidates: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Drop offerings another offering beats on strength *and* price.

    Ties are kept: two models at the same price and strength are equally valid
    choices, and dropping one arbitrarily would hide a real option.
    """
    return [
        c
        for c in candidates
        if not any(
            o is not c and o[1] >= c[1] and o[2] <= c[2] and (o[1] > c[1] or o[2] < c[2])
            for o in candidates
        )
    ]


def build_personas(offerings: list[dict], strength: dict[str, float]) -> dict[str, list[str]]:
    """``persona -> provider model ids, best first`` for one provider's lineup.

    ``offerings`` are the provider's catalog offerings (``canonical_id``,
    ``provider_model_id``, ``pricing``); ``strength`` maps canonical id -> Elo.
    Offerings with no Elo or no complete price are skipped — a preference order
    is only worth emitting where both axes are real.
    """
    result: dict[str, list[str]] = {}
    for persona in _PROFILES:
        candidates = []
        for offering in offerings:
            elo = strength.get(offering.get("canonical_id"))
            cost = _blended_cost(offering.get("pricing"), persona.input_share)
            if elo is None or cost is None:
                continue
            candidates.append((offering["provider_model_id"], elo, cost))
        if not candidates:
            continue
        candidates = _frontier(candidates)
        scale_elo = _scale([c[1] for c in candidates])
        scale_cost = _scale([c[2] for c in candidates])
        weight = persona.strength_weight

        def score(c, _e=scale_elo, _c=scale_cost, _w=weight):
            # Cost enters negated: cheaper is better, and a persona that weights
            # strength at 0.9 spends only 0.1 of its score caring about price.
            return _w * _e(c[1]) - (1.0 - _w) * _c(c[2])

        # Sort by score, then cheaper, then by id so equal candidates are stable.
        result[persona.name] = [
            c[0] for c in sorted(candidates, key=lambda c: (-score(c), c[2], c[0]))
        ]
    return result

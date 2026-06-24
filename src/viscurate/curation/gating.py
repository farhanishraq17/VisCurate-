"""Verifier gating: a relation *fact* → a structural-edit *permission* (CLAUDE.md §3.5.7).

The verifier answers "what *is* the relation?"; the agent proposes "what *should* be done?".
A structural edit (``merge`` / ``parameterize``) **cannot proceed without a certifying relation**
(CLAUDE.md §3.2). This module is that hard gate — it maps a :class:`RelationResult` to a
permit/deny decision plus the **structured, actionable rejection** of CLAUDE.md §3.5.7 (relation,
deciding distances, and the alternatives the relation still permits). It never mutates the
library and never sees a skill's description — it consumes only the verifier's output.

The relation → permitted-action map (CLAUDE.md §3.5.7):

======================  ===========================================
 relation                permitted structural action
======================  ===========================================
 EXACT / PERCEPTUAL       merge (to one canonical)
 SUBSUMPTION (A ⊑ B)      parameterize (fold the specialization A into the generalizer B)
 SEMANTIC_PRESERVING      parameterize / unify (or keep both)
 COMPLEMENTARY / DISTINCT (no structural edit — keep separate)
 UNCERTAIN                (no structural edit — route to human review)
======================  ===========================================

The agent still decides what is *desirable* using usage/budget (e.g. declining a permitted
subsumption fold because the specialization is heavily used) — that gate lives in the
environment/agent, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from viscurate.curation.actions import Action, ActionKind
from viscurate.equivalence.relations import Direction, Relation, RelationResult

__all__ = ["GateDecision", "gate_structural"]


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict for one structural action, carrying the verifier's evidence."""

    permitted: bool
    reason: str
    relation: Relation
    direction: Direction = Direction.NONE
    distances: dict[str, float] = field(default_factory=dict)
    alternatives: tuple[str, ...] = ()


def _deny(result: RelationResult, *, alternatives: tuple[str, ...] | None = None) -> GateDecision:
    return GateDecision(
        permitted=False,
        reason=f"{result.relation.value}: {result.reason}",
        relation=result.relation,
        direction=result.direction,
        distances=dict(result.distances),
        alternatives=alternatives if alternatives is not None else result.alternatives,
    )


def _allow(result: RelationResult) -> GateDecision:
    return GateDecision(
        permitted=True,
        reason=f"{result.relation.value}: {result.reason}",
        relation=result.relation,
        direction=result.direction,
        distances=dict(result.distances),
        alternatives=result.alternatives,
    )


def gate_structural(action: Action, result: RelationResult) -> GateDecision:
    """Decide whether ``result`` (the verifier's relation for ``(primary, secondary)``) licenses
    ``action``. ``classify`` must be called with ``view_a = primary``, ``view_b = secondary`` so
    the subsumption direction is interpreted relative to the action's operands.
    """
    if action.kind is ActionKind.MERGE:
        if result.licenses_merge:
            return _allow(result)
        return _deny(result)

    if action.kind is ActionKind.PARAMETERIZE:
        if result.relation is Relation.SUBSUMPTION:
            # primary ⊑ secondary == B_SUBSUMES_A (A=primary, B=secondary). Fold the
            # specialization into the generalizer; never the reverse (CLAUDE.md §3.5.4).
            if result.direction is Direction.B_SUBSUMES_A:
                return _allow(result)
            return _deny(
                result,
                alternatives=(
                    "parameterize (swap operands: fold secondary into primary)",
                    "keep_separate",
                ),
            )
        if result.relation is Relation.SEMANTIC_PRESERVING:
            return _allow(result)
        return _deny(result)

    # SPLIT and any non-structural kind are not relation-gated here (CLAUDE.md leaves split's
    # certifying relation to the agent-authored pieces, which are untrusted in v1). The
    # environment routes those; reaching this point is a programming error.
    raise ValueError(f"gate_structural called on non-gated action kind {action.kind!r}")

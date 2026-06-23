"""The relation vocabulary and the structured verifier result (CLAUDE.md §3.5.3, §3.5.7).

Six relations partition every ordered pair ``(A, B)``, with an explicit **UNCERTAIN**
abstention class for pairs that land inside a calibrated band around a threshold. The
verifier returns a :class:`RelationResult` — a *fact about outputs* carrying the deciding
distances and the worst-case probe — never a curation decision (that is the agent's job).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Relation(StrEnum):
    """The six output-grounded relations plus the abstention class (CLAUDE.md §3.5.3).

    Ordered by the stop-at-first pipeline: cheapest-and-strictest first. ``UNCERTAIN`` is not
    a relation but an honest "calibration-noise" verdict for band-straddling pairs.
    """

    EXACT = "EXACT"
    PERCEPTUAL = "PERCEPTUAL"
    SUBSUMPTION = "SUBSUMPTION"
    SEMANTIC_PRESERVING = "SEMANTIC_PRESERVING"
    COMPLEMENTARY = "COMPLEMENTARY"
    DISTINCT = "DISTINCT"
    UNCERTAIN = "UNCERTAIN"


class Direction(StrEnum):
    """Direction of a directional relation (subsumption). ``NONE`` for symmetric relations."""

    NONE = "NONE"
    A_SUBSUMES_B = "A_SUBSUMES_B"  # B ⊑ A : every B-binding is reproduced by some A-binding
    B_SUBSUMES_A = "B_SUBSUMES_A"  # A ⊑ B : every A-binding is reproduced by some B-binding


@dataclass(frozen=True)
class RelationResult:
    """The verifier's answer for one pair: the relation, its direction, and the evidence.

    ``distances`` records the deciding metric values (``l_inf``, ``lpips``, ``ssim_dist``,
    ``dino`` quantiles, ``commute`` …) so a rejected merge carries an actionable reason
    (CLAUDE.md §3.5.7). ``worst_probe`` is the probe id that drove a worst-case verdict.
    ``alternatives`` lists structural actions the relation still permits.
    """

    relation: Relation
    direction: Direction = Direction.NONE
    reason: str = ""
    distances: dict[str, float] = field(default_factory=dict)
    worst_probe: str = ""
    alternatives: tuple[str, ...] = ()
    uncertain_about: Relation | None = None

    @property
    def is_uncertain(self) -> bool:
        return self.relation is Relation.UNCERTAIN

    @property
    def licenses_merge(self) -> bool:
        """EXACT/PERCEPTUAL license a merge (CLAUDE.md §3.5.7 relation→action map)."""
        return self.relation in (Relation.EXACT, Relation.PERCEPTUAL)

    @property
    def licenses_parameterize(self) -> bool:
        """SUBSUMPTION/SEMANTIC license parameterize/unify."""
        return self.relation in (Relation.SUBSUMPTION, Relation.SEMANTIC_PRESERVING)

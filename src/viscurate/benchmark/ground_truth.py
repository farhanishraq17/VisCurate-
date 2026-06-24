"""The designed relation graph ``G0`` — the Phase-4 answer key (CLAUDE.md §1.1, §2.3, §2.5).

``G0`` lifts the blueprint's hand-authored ``known_equivalences`` into a machine-readable,
validated graph over ordered pairs. Both the output-grounded verifier and the text baselines
are scored against it; it is **fixed before any metric runs** and is never derived from the
metrics under test (CLAUDE.md §2.5).

Enforced graph properties (CLAUDE.md §2.3):

* **subsumption is a DAG** — ``spec ⊑ gen`` edges contain no cycle;
* **EXACT is transitive** — its connected components are closure-checked (trivially satisfied
  in clean ``L0``, where EXACT is empty);
* **PERCEPTUAL / SEMANTIC / COMPLEMENTARY are symmetric but NOT transitive** — stored
  unordered, queried order-independently;
* a pair may carry **only one** designed relation (no pair appears in two categories).

Any pair not listed is :data:`~viscurate.equivalence.relations.Relation.DISTINCT` by default
(the ``O(N²)`` distinct sea). Engineered hard negatives are DISTINCT pairs *flagged* so the
benchmark can report that slice separately — it is where text judges fail and the contribution
lives.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from viscurate.equivalence.candidates import normalize_pair
from viscurate.equivalence.relations import Direction, Relation

__all__ = [
    "DesignedLabel",
    "GroundTruthGraph",
    "GroundTruthSpec",
    "load_ground_truth",
]


@dataclass(frozen=True)
class DesignedLabel:
    """The designed relation for one ordered query pair ``(a, b)``.

    ``direction`` is meaningful only for SUBSUMPTION and is expressed **relative to the query
    order** using the :class:`~viscurate.equivalence.relations.Direction` convention
    (``B_SUBSUMES_A`` == ``a ⊑ b``). ``is_hard_negative`` flags an engineered DISTINCT pair.
    """

    relation: Relation
    direction: Direction = Direction.NONE
    is_hard_negative: bool = False

    @property
    def mergeable(self) -> bool:
        """Whether the *truth* licenses a merge (EXACT/PERCEPTUAL) — the binary decision axis."""
        return self.relation in (Relation.EXACT, Relation.PERCEPTUAL)


class _SubsumptionEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spec: str  # the specialization (subsumed)
    gen: str  # the generalization (subsumer)


class GroundTruthSpec(BaseModel):
    """The on-disk shape of ``configs/ground_truth_g0.yaml`` (validated)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "1"
    exact: tuple[tuple[str, str], ...] = ()
    perceptual: tuple[tuple[str, str], ...] = ()
    subsumption: tuple[_SubsumptionEdge, ...] = ()
    semantic_preserving: tuple[tuple[str, str], ...] = ()
    complementary: tuple[tuple[str, str], ...] = ()
    distinct_hard_negatives: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def _no_self_pairs(self) -> GroundTruthSpec:
        for a, b in (
            *self.exact,
            *self.perceptual,
            *self.semantic_preserving,
            *self.complementary,
            *self.distinct_hard_negatives,
        ):
            if a == b:
                raise ValueError(f"G0: self-pair not allowed: ({a}, {b})")
        for e in self.subsumption:
            if e.spec == e.gen:
                raise ValueError(f"G0: self subsumption not allowed: {e.spec}")
        return self


class GroundTruthGraph:
    """A validated, queryable designed relation graph.

    Construct via :func:`load_ground_truth`. The lookup :meth:`label` returns a
    :class:`DesignedLabel` for any ordered pair, defaulting to DISTINCT.
    """

    def __init__(self, spec: GroundTruthSpec, *, valid_ids: Iterable[str] | None = None) -> None:
        self.version = spec.version
        # symmetric relations: normalized pair -> Relation
        self._symmetric: dict[tuple[str, str], Relation] = {}
        # subsumption: normalized pair -> (spec_id, gen_id)
        self._subsumption: dict[tuple[str, str], tuple[str, str]] = {}
        # engineered hard negatives (DISTINCT), normalized
        self._hard_negatives: set[tuple[str, str]] = set()
        self._build(spec)
        if valid_ids is not None:
            self._validate_ids(set(valid_ids))
        self._validate_subsumption_dag()
        self._validate_exact_closure()

    # -- construction -----------------------------------------------------------------
    def _claim(self, a: str, b: str, where: str) -> tuple[str, str]:
        key = normalize_pair(a, b)
        if key in self._symmetric or key in self._subsumption:
            raise ValueError(f"G0: pair {key} listed more than once (at {where})")
        return key

    def _build(self, spec: GroundTruthSpec) -> None:
        for rel, pairs in (
            (Relation.EXACT, spec.exact),
            (Relation.PERCEPTUAL, spec.perceptual),
            (Relation.SEMANTIC_PRESERVING, spec.semantic_preserving),
            (Relation.COMPLEMENTARY, spec.complementary),
        ):
            for a, b in pairs:
                self._symmetric[self._claim(a, b, rel.value)] = rel
        for e in spec.subsumption:
            self._subsumption[self._claim(e.spec, e.gen, "subsumption")] = (e.spec, e.gen)
        for a, b in spec.distinct_hard_negatives:
            key = self._claim(a, b, "distinct_hard_negatives")
            self._symmetric[key] = Relation.DISTINCT
            self._hard_negatives.add(key)

    # -- validation -------------------------------------------------------------------
    def _validate_ids(self, valid: set[str]) -> None:
        missing = sorted(i for i in self.skill_ids() if i not in valid)
        if missing:
            raise ValueError(f"G0 references unknown skill ids: {missing}")

    def _validate_subsumption_dag(self) -> None:
        # Edge spec -> gen; a cycle would make "specialization" ill-defined (CLAUDE.md §2.3).
        adj: dict[str, list[str]] = {}
        for spec_id, gen_id in self._subsumption.values():
            adj.setdefault(spec_id, []).append(gen_id)
        visiting, done = set(), set()

        def visit(node: str, stack: tuple[str, ...]) -> None:
            if node in done:
                return
            if node in visiting:
                cycle = " -> ".join((*stack, node))
                raise ValueError(f"G0: subsumption is not a DAG (cycle: {cycle})")
            visiting.add(node)
            for nxt in adj.get(node, ()):
                visit(nxt, (*stack, node))
            visiting.discard(node)
            done.add(node)

        for node in list(adj):
            visit(node, ())

    def _validate_exact_closure(self) -> None:
        # EXACT must be transitive: if A=B and B=C are listed, A=C must be too (CLAUDE.md §2.3).
        exact_pairs = [k for k, r in self._symmetric.items() if r is Relation.EXACT]
        adj: dict[str, set[str]] = {}
        for a, b in exact_pairs:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        # Build connected components; every within-component pair must be present.
        seen: set[str] = set()
        for start in list(adj):
            if start in seen:
                continue
            comp: list[str] = []
            stack = [start]
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                comp.append(n)
                stack.extend(adj.get(n, ()))
            for i in range(len(comp)):
                for j in range(i + 1, len(comp)):
                    if normalize_pair(comp[i], comp[j]) not in self._symmetric:
                        raise ValueError(
                            f"G0: EXACT is not transitively closed; missing ({comp[i]}, {comp[j]})"
                        )

    # -- queries ----------------------------------------------------------------------
    def label(self, a: str, b: str) -> DesignedLabel:
        """The designed relation for ordered pair ``(a, b)`` (DISTINCT if unlisted)."""
        key = normalize_pair(a, b)
        sub = self._subsumption.get(key)
        if sub is not None:
            spec_id, _gen_id = sub
            # Direction relative to the QUERY order: a ⊑ b == B_SUBSUMES_A.
            direction = Direction.B_SUBSUMES_A if a == spec_id else Direction.A_SUBSUMES_B
            return DesignedLabel(Relation.SUBSUMPTION, direction)
        rel = self._symmetric.get(key)
        if rel is not None:
            return DesignedLabel(rel, Direction.NONE, is_hard_negative=key in self._hard_negatives)
        return DesignedLabel(Relation.DISTINCT, Direction.NONE)

    def is_hard_negative(self, a: str, b: str) -> bool:
        return normalize_pair(a, b) in self._hard_negatives

    def designed_pairs(self) -> set[tuple[str, str]]:
        """Every explicitly-listed pair (non-default), normalized — the planted structure.

        The benchmark unions these into the candidate set so the planted relations are always
        scored even if output-based candidate generation does not surface them.
        """
        return set(self._symmetric) | set(self._subsumption)

    def skill_ids(self) -> set[str]:
        ids: set[str] = set()
        for a, b in self.designed_pairs():
            ids.add(a)
            ids.add(b)
        return ids

    def labels_for(self, pairs: Sequence[tuple[str, str]]) -> dict[tuple[str, str], DesignedLabel]:
        """Designed labels for a set of (already-normalized or raw) pairs."""
        return {normalize_pair(a, b): self.label(a, b) for a, b in pairs}


def load_ground_truth(
    path: str | Path, *, valid_ids: Iterable[str] | None = None
) -> GroundTruthGraph:
    """Load and validate ``G0`` from YAML; optionally check ids against the live library."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    spec = GroundTruthSpec.model_validate(raw)
    return GroundTruthGraph(spec, valid_ids=valid_ids)

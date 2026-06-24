"""Human-verification infrastructure for the judgment-laden slice (CLAUDE.md §2.5, Phase 4).

PERCEPTUAL-borderline, SEMANTIC_PRESERVING, and SUBSUMPTION relations are intrinsically
judgment-laden, so the dataset routes them to **human verification** with a reported
inter-annotator agreement (κ). This module:

* **extracts the review slice** — the designed SEMANTIC/SUBSUMPTION pairs plus any pair the
  output verifier returned as UNCERTAIN (its abstention band is exactly the review queue);
* **writes a labeling template** (JSON) annotators fill with one relation per pair;
* **loads completed labels** (one file per annotator) and computes **Cohen's / Fleiss' κ**.

It does **not** invent labels or a κ value. With no completed annotations the agreement is
reported with ``status="pending"`` — "not yet run" is acceptable; fabricated numbers are not
(CLAUDE.md §5).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from viscurate.equivalence.relations import Relation

if TYPE_CHECKING:  # avoid an import cycle (runner imports nothing from here)
    from viscurate.benchmark.runner import BenchmarkResult
    from viscurate.skills.model import SkillSpec

__all__ = [
    "KappaResult",
    "ReviewItem",
    "cohen_kappa",
    "extract_review_slice",
    "fleiss_kappa",
    "inter_annotator_agreement",
    "load_review_labels",
    "write_review_template",
]

Pair = tuple[str, str]

# Relations a human is asked to confirm (the judgment-laden ones, CLAUDE.md §2.5).
_REVIEW_RELATIONS = (Relation.PERCEPTUAL, Relation.SEMANTIC_PRESERVING, Relation.SUBSUMPTION)


@dataclass(frozen=True)
class ReviewItem:
    """One pair queued for human verification, with the text + machine verdicts for context."""

    pair: Pair
    a_name: str
    b_name: str
    a_description: str
    b_description: str
    designed_relation: Relation
    output_relation: Relation
    output_reason: str

    def to_template(self) -> dict[str, object]:
        return {
            "pair": list(self.pair),
            "a_name": self.a_name,
            "b_name": self.b_name,
            "a_description": self.a_description,
            "b_description": self.b_description,
            "designed_relation": self.designed_relation.value,
            "output_relation": self.output_relation.value,
            "output_reason": self.output_reason,
            "label": "",  # annotator fills one of the Relation names
        }


def extract_review_slice(
    result: BenchmarkResult, spec_by_id: Mapping[str, SkillSpec]
) -> list[ReviewItem]:
    """Pairs needing human verification: designed SEMANTIC/SUBSUMPTION/PERCEPTUAL ∪ UNCERTAIN."""
    items: list[ReviewItem] = []
    for (a_id, b_id), label in result.truth.items():
        out = result.output_track.predictions.get((a_id, b_id))
        needs = label.relation in _REVIEW_RELATIONS or (out is not None and out.uncertain)
        if not needs:
            continue
        a, b = spec_by_id[a_id], spec_by_id[b_id]
        items.append(
            ReviewItem(
                pair=(a_id, b_id),
                a_name=a.name,
                b_name=b.name,
                a_description=a.description,
                b_description=b.description,
                designed_relation=label.relation,
                output_relation=out.relation if out is not None else Relation.DISTINCT,
                output_reason=out.reason if out is not None else "",
            )
        )
    items.sort(key=lambda it: it.pair)
    return items


def write_review_template(items: Sequence[ReviewItem], path: str | Path) -> Path:
    """Write the labeling template (JSON). Annotators copy it and fill each ``label`` field."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instructions": (
            "For each pair, set `label` to exactly one relation name "
            f"({', '.join(r.value for r in Relation if r is not Relation.UNCERTAIN)}). "
            "Judge from the descriptions and your understanding of the operations; leave blank "
            "if genuinely unsure. One file per annotator."
        ),
        "items": [it.to_template() for it in items],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_review_labels(paths: Sequence[str | Path]) -> dict[Pair, list[Relation]]:
    """Load filled templates (one per annotator) → ``pair -> [relation per annotator]``.

    Only non-empty, valid labels are kept. Pairs are aligned across annotators by id.
    """
    per_pair: dict[Pair, list[Relation]] = {}
    valid = {r.value for r in Relation}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in data.get("items", []):
            raw = str(item.get("label", "")).strip().upper()
            if raw not in valid or raw == Relation.UNCERTAIN.value:
                continue
            a, b = item["pair"]
            per_pair.setdefault((a, b), []).append(Relation(raw))
    return per_pair


@dataclass(frozen=True)
class KappaResult:
    """Inter-annotator agreement, or an honest 'pending' when there is nothing to compute."""

    status: str  # "computed" | "pending"
    kappa: float | None
    n_items: int
    n_annotators: int
    note: str = ""


def cohen_kappa(a: Sequence[Relation], b: Sequence[Relation]) -> float:
    """Cohen's κ for two annotators over aligned categorical labels."""
    if len(a) != len(b):
        raise ValueError("annotator label sequences must be the same length")
    n = len(a)
    if n == 0:
        raise ValueError("no items to score")
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    po = agree / n
    cats = set(a) | set(b)
    pe = 0.0
    for c in cats:
        pa = sum(1 for x in a if x == c) / n
        pb = sum(1 for y in b if y == c) / n
        pe += pa * pb
    if pe >= 1.0:  # perfect chance agreement (one category) → κ undefined; treat as 1.0
        return 1.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(ratings: Sequence[Sequence[Relation]]) -> float:
    """Fleiss' κ for ≥3 annotators. ``ratings[i]`` is the labels assigned to item ``i``.

    Every item must have the same number of annotators (the standard Fleiss assumption).
    """
    if not ratings:
        raise ValueError("no items to score")
    n_raters = len(ratings[0])
    if any(len(r) != n_raters for r in ratings):
        raise ValueError("Fleiss' κ requires the same number of annotators per item")
    if n_raters < 2:
        raise ValueError("need at least 2 annotators per item")
    categories = sorted({c for r in ratings for c in r}, key=lambda c: c.value)
    n_items = len(ratings)
    # P_i = agreement within item i.
    p_i = []
    cat_totals = dict.fromkeys(categories, 0)
    for r in ratings:
        counts = dict.fromkeys(categories, 0)
        for c in r:
            counts[c] += 1
            cat_totals[c] += 1
        p_i.append((sum(v * v for v in counts.values()) - n_raters) / (n_raters * (n_raters - 1)))
    p_bar = sum(p_i) / n_items
    p_e = sum((tot / (n_items * n_raters)) ** 2 for tot in cat_totals.values())
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def inter_annotator_agreement(per_pair: Mapping[Pair, list[Relation]]) -> KappaResult:
    """Compute κ over the items every annotator labeled, or report ``pending`` if not possible.

    Uses Cohen's κ for exactly two annotators and Fleiss' κ for three or more. Only items with
    the full set of annotators are scored (so the agreement is over a common, aligned slice).
    """
    if not per_pair:
        return KappaResult("pending", None, 0, 0, "no completed annotations yet")
    n_annotators = max((len(v) for v in per_pair.values()), default=0)
    if n_annotators < 2:
        return KappaResult("pending", None, len(per_pair), n_annotators, "need ≥2 annotators for κ")
    aligned = {pair: labs for pair, labs in per_pair.items() if len(labs) == n_annotators}
    if not aligned:
        return KappaResult("pending", None, 0, n_annotators, "no item labeled by all annotators")
    items = sorted(aligned)
    if n_annotators == 2:
        a = [aligned[p][0] for p in items]
        b = [aligned[p][1] for p in items]
        return KappaResult("computed", cohen_kappa(a, b), len(items), 2, "Cohen's κ")
    ratings = [aligned[p] for p in items]
    return KappaResult("computed", fleiss_kappa(ratings), len(items), n_annotators, "Fleiss' κ")

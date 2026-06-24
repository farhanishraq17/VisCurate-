"""Benchmark metrics (CLAUDE.md §3.5.9) — pure functions over predictions vs the answer key.

Three families of number come out of Phase 4:

* **per-relation precision / recall / F1** and a 6×6 **confusion matrix** per judge track —
  how well each track recovers the fine taxonomy (the output verifier is the only track that
  can resolve all six; text judges resolve at most merge-vs-distinct);
* the **mergeable** decision (EXACT ∪ PERCEPTUAL) scored as one-vs-rest — the binary axis
  every track shares, and the **false-merge / precision-on-DISTINCT** safety numbers (a false
  merge costs more than a missed compression, CLAUDE.md §3.5.5);
* the headline **divergence statistic** — the rate at which a text track and the output track
  disagree on the mergeable decision, broken down by the *true* relation, with the engineered
  hard-negative slice reported separately (CLAUDE.md §3.5.9).

Everything here is stdlib + numpy and free of ML / I/O, so it is deterministically testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from viscurate.benchmark.ground_truth import DesignedLabel
from viscurate.benchmark.tracks import Verdict
from viscurate.equivalence.relations import Relation

__all__ = [
    "PRF",
    "DivergenceRow",
    "SafetyStats",
    "abstention_rate",
    "confusion_matrix",
    "divergence_by_true_relation",
    "mergeable_prf",
    "per_relation_prf",
    "precision_on_distinct",
]

# Display/iteration order for the fine taxonomy (UNCERTAIN is a predicted-only column).
RELATION_ORDER: tuple[Relation, ...] = (
    Relation.EXACT,
    Relation.PERCEPTUAL,
    Relation.SUBSUMPTION,
    Relation.SEMANTIC_PRESERVING,
    Relation.COMPLEMENTARY,
    Relation.DISTINCT,
)

Pair = tuple[str, str]
Truth = Mapping[Pair, DesignedLabel]
Preds = Mapping[Pair, Verdict]


@dataclass(frozen=True)
class PRF:
    """Precision / recall / F1 for one class, with the raw counts behind them."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    @property
    def support(self) -> int:
        return self.tp + self.fn


def _prf(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def _common(true: Truth, pred: Preds) -> list[Pair]:
    """Pairs scored by both — the only ones a metric may use (sorted for determinism)."""
    return sorted(set(true) & set(pred))


def per_relation_prf(true: Truth, pred: Preds) -> dict[Relation, PRF]:
    """One-vs-rest P/R/F1 for each of the six relations (predicted UNCERTAIN counts as abstain)."""
    pairs = _common(true, pred)
    out: dict[Relation, PRF] = {}
    for rel in RELATION_ORDER:
        tp = fp = fn = 0
        for key in pairs:
            t = true[key].relation is rel
            p = pred[key].relation is rel  # UNCERTAIN never equals a real relation → abstain
            if t and p:
                tp += 1
            elif p and not t:
                fp += 1
            elif t and not p:
                fn += 1
        out[rel] = _prf(tp, fp, fn)
    return out


def confusion_matrix(true: Truth, pred: Preds) -> dict[Relation, dict[Relation, int]]:
    """``matrix[true_relation][predicted_relation] -> count`` (UNCERTAIN included as predicted)."""
    cols = (*RELATION_ORDER, Relation.UNCERTAIN)
    matrix: dict[Relation, dict[Relation, int]] = {
        t: dict.fromkeys(cols, 0) for t in RELATION_ORDER
    }
    for key in _common(true, pred):
        matrix[true[key].relation][pred[key].relation] += 1
    return matrix


def mergeable_prf(true: Truth, pred: Preds) -> PRF:
    """P/R/F1 with the positive class = *mergeable* (EXACT ∪ PERCEPTUAL).

    Low precision here means many **false merges** — the failure mode the project exists to
    prevent. This is the axis on which every track (output and text) is directly comparable.
    """
    tp = fp = fn = 0
    for key in _common(true, pred):
        t, p = true[key].mergeable, pred[key].mergeable
        if t and p:
            tp += 1
        elif p and not t:
            fp += 1
        elif t and not p:
            fn += 1
    return _prf(tp, fp, fn)


@dataclass(frozen=True)
class SafetyStats:
    """The safety-critical numbers on truly-DISTINCT pairs (CLAUDE.md §3.5.9)."""

    n_distinct: int
    false_merges: int  # truly DISTINCT but predicted mergeable

    @property
    def false_merge_rate(self) -> float:
        return self.false_merges / self.n_distinct if self.n_distinct else 0.0

    @property
    def precision_on_distinct(self) -> float:
        """Fraction of truly-DISTINCT pairs correctly kept separate (1 − false-merge rate)."""
        return 1.0 - self.false_merge_rate


def precision_on_distinct(
    true: Truth, pred: Preds, *, hard_negatives_only: bool = False
) -> SafetyStats:
    """How often a track wrongly licenses a merge on truly-DISTINCT pairs.

    With ``hard_negatives_only`` restrict to the engineered hard-negative slice — where the
    contribution lives (text judges merge them; the output judge must not).
    """
    n = fm = 0
    for key in _common(true, pred):
        label = true[key]
        if label.relation is not Relation.DISTINCT:
            continue
        if hard_negatives_only and not label.is_hard_negative:
            continue
        n += 1
        if pred[key].mergeable:
            fm += 1
    return SafetyStats(n_distinct=n, false_merges=fm)


def abstention_rate(pred: Preds) -> float:
    """Fraction of a track's predictions that abstain (UNCERTAIN); 0 for non-abstaining tracks."""
    if not pred:
        return 0.0
    return sum(1 for v in pred.values() if v.uncertain) / len(pred)


@dataclass(frozen=True)
class DivergenceRow:
    """Output-vs-text disagreement on the mergeable decision for one true-relation slice."""

    slice_name: str
    n: int
    disagree: int
    text_over_merge: int  # text says merge, output does not (the silent-merge danger)
    text_under_merge: int  # output says merge, text does not (redundancy text misses)

    @property
    def disagree_rate(self) -> float:
        return self.disagree / self.n if self.n else 0.0


def divergence_by_true_relation(
    output_pred: Preds, text_pred: Preds, true: Truth
) -> list[DivergenceRow]:
    """Disagreement between the output track and one text track, per true relation + slices.

    Rows: one per relation present, a separate ``hard_negative`` slice, and ``ALL``. The
    project's premise requires this to be non-zero — especially on the hard-negative slice
    (CLAUDE.md Phase 4 go/no-go).
    """
    keys = sorted(set(output_pred) & set(text_pred) & set(true))

    def row(name: str, subset: list[Pair]) -> DivergenceRow:
        disagree = over = under = 0
        for k in subset:
            o, t = output_pred[k].mergeable, text_pred[k].mergeable
            if o != t:
                disagree += 1
                if t and not o:
                    over += 1
                else:
                    under += 1
        return DivergenceRow(name, len(subset), disagree, over, under)

    rows: list[DivergenceRow] = []
    for rel in RELATION_ORDER:
        subset = [k for k in keys if true[k].relation is rel]
        if subset:
            rows.append(row(rel.value, subset))
    hard = [k for k in keys if true[k].is_hard_negative]
    if hard:
        rows.append(row("hard_negative", hard))
    rows.append(row("ALL", keys))
    return rows

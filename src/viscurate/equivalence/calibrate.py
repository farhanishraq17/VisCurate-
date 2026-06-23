"""Threshold calibration (CLAUDE.md §3.5.5).

ε, τ_perc, τ_sem and the abstention band δ are **calibrated, not guessed**, on the
human-labeled validation split to maximize **precision on non-equivalence** (a false merge
costs more than a missed compression) subject to a recall floor. The procedure here is
distance-only and deterministic; the labeled split itself is produced in Phase 4 (with human
verification of the PERCEPTUAL/SEMANTIC slice). Calibration is cluster-disjoint from test and
frozen before any test metric runs.

This module implements the *procedure* and the provenance stamp (split hash + date written into
the config so a reported metric can never use uncalibrated thresholds). It does **not** ship
calibrated numbers — there is no labeled split yet, and inventing one is forbidden (CLAUDE.md
§5: "Not yet run" is acceptable; fabricated numbers are not).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from viscurate.config import ThresholdConfig
from viscurate.equivalence.relations import Relation

__all__ = [
    "LabeledPair",
    "ThresholdFit",
    "calibrate_thresholds",
    "select_threshold",
    "split_hash",
]

# A pair counts as "mergeable" (the equivalence the precision target protects) iff truly
# EXACT or PERCEPTUAL; "semantic-or-closer" additionally includes SEMANTIC_PRESERVING.
_MERGEABLE = (Relation.EXACT, Relation.PERCEPTUAL)
_SEMANTIC_OR_CLOSER = (Relation.EXACT, Relation.PERCEPTUAL, Relation.SEMANTIC_PRESERVING)


@dataclass(frozen=True)
class LabeledPair:
    """One validation pair: its computed distances and the human-verified true relation."""

    true: Relation
    distances: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdFit:
    """A selected threshold and the operating point it achieves on the split."""

    threshold: float
    precision: float
    recall: float
    n_positive: int
    n_negative: int
    met_target: bool


def split_hash(pairs: Sequence[LabeledPair]) -> str:
    """Deterministic content hash of a labeled split (for the calibration provenance stamp)."""
    h = hashlib.sha256()
    for p in pairs:
        h.update(p.true.value.encode("ascii"))
        h.update(json.dumps(p.distances, sort_keys=True, separators=(",", ":")).encode("ascii"))
        h.update(b"|")
    return h.hexdigest()


def select_threshold(
    samples: Sequence[tuple[float, bool]],
    *,
    min_precision: float,
    min_recall: float,
) -> ThresholdFit:
    """Pick the most permissive threshold (predict positive iff ``score ≤ t``) that keeps
    precision ≥ ``min_precision`` and recall ≥ ``min_recall``; else the most precise fallback.

    Maximizing recall subject to a precision floor is exactly "compress as much as possible
    without licensing a false merge" (CLAUDE.md §3.5.5).
    """
    pos_total = sum(1 for _, y in samples if y)
    candidates = sorted({s for s, _ in samples})
    # Include a below-all threshold so "predict nothing positive" is an option.
    thresholds = [float("-inf"), *candidates]

    best: ThresholdFit | None = None
    fallback: ThresholdFit | None = None
    for t in thresholds:
        tp = sum(1 for s, y in samples if s <= t and y)
        fp = sum(1 for s, y in samples if s <= t and not y)
        predicted = tp + fp
        precision = 1.0 if predicted == 0 else tp / predicted
        recall = 0.0 if pos_total == 0 else tp / pos_total
        fit = ThresholdFit(t, precision, recall, pos_total, len(samples) - pos_total, False)
        meets = precision >= min_precision and recall >= min_recall
        if meets and (best is None or fit.recall > best.recall):
            best = ThresholdFit(t, precision, recall, fit.n_positive, fit.n_negative, True)
        # Fallback: highest precision, then highest recall (the conservative choice).
        if fallback is None or (precision, recall) > (fallback.precision, fallback.recall):
            fallback = fit
    result = best if best is not None else fallback
    assert result is not None  # thresholds always has at least one element
    return result


def calibrate_thresholds(
    pairs: Sequence[LabeledPair],
    *,
    base: ThresholdConfig,
    date: str,
    min_precision: float = 0.99,
    min_recall: float = 0.5,
    delta_grid: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
) -> tuple[ThresholdConfig, dict[str, ThresholdFit]]:
    """Calibrate τ_perc (LPIPS) and τ_sem (DINO p90) on a labeled split; stamp provenance.

    ε is left at its rounding default (it is a numerical-equality tolerance, not a learned
    operating point). δ is the smallest band in ``delta_grid`` that lifts decisive-pair
    precision to ``min_precision`` for the PERCEPTUAL threshold. Returns the calibrated config
    plus the per-threshold fits for the calibration report.
    """
    perc_samples = [
        (p.distances["lpips"], p.true in _MERGEABLE) for p in pairs if "lpips" in p.distances
    ]
    sem_samples = [
        (p.distances["dino_p90"], p.true in _SEMANTIC_OR_CLOSER)
        for p in pairs
        if "dino_p90" in p.distances
    ]
    fits: dict[str, ThresholdFit] = {}
    tau_perc = base.perceptual_lpips
    tau_sem = base.semantic_dino
    if perc_samples:
        fits["perceptual_lpips"] = select_threshold(
            perc_samples, min_precision=min_precision, min_recall=min_recall
        )
        tau_perc = fits["perceptual_lpips"].threshold
    if sem_samples:
        fits["semantic_dino"] = select_threshold(
            sem_samples, min_precision=min_precision, min_recall=min_recall
        )
        tau_sem = fits["semantic_dino"].threshold

    delta = _select_delta(perc_samples, tau_perc, delta_grid, min_precision)

    calibrated = base.model_copy(
        update={
            "perceptual_lpips": _finite(tau_perc, base.perceptual_lpips),
            "semantic_dino": _finite(tau_sem, base.semantic_dino),
            "abstention_delta": delta,
            "calibrated": True,
            "calibration_split_hash": split_hash(pairs),
            "calibration_date": date,
        }
    )
    return calibrated, fits


def _finite(value: float, fallback: float) -> float:
    """A ``-inf`` "predict nothing" threshold is unusable as a config value → keep the base."""
    return value if value not in (float("inf"), float("-inf")) else fallback


def _select_delta(
    perc_samples: Sequence[tuple[float, bool]],
    tau: float,
    delta_grid: Sequence[float],
    min_precision: float,
) -> float:
    """Smallest δ whose abstention band lifts decisive-pair precision to ``min_precision``."""
    if not perc_samples or tau in (float("inf"), float("-inf")):
        return delta_grid[0]
    for delta in sorted(delta_grid):
        lo, hi = tau * (1.0 - delta), tau * (1.0 + delta)
        decisive = [(s, y) for s, y in perc_samples if not (lo <= s <= hi)]
        tp = sum(1 for s, y in decisive if s <= tau and y)
        fp = sum(1 for s, y in decisive if s <= tau and not y)
        precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
        if precision >= min_precision:
            return delta
    return max(delta_grid)

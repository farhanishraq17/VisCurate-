"""The output-grounded equivalence engine (CLAUDE.md §3, §3.5 — Phase 3).

This package decides, *from executed outputs alone*, which of six relations holds between a
pair of skills, and exposes the per-relation detectors, the stop-at-first taxonomy with its
UNCERTAIN abstention band, directional subsumption search, the COMPLEMENTARY composition
test, output-based candidate generation, and the threshold-calibration procedure.

**The load-bearing modality boundary (CLAUDE.md §1.2) is enforced by type here.** Nothing in
this package reads a skill's ``description``. The taxonomy is handed a
:class:`~viscurate.skills.model.ComparatorView` (no ``description`` attribute) plus an
:class:`~viscurate.equivalence.compare.OutputProvider` (yields *outputs*, never text). The ML
backends (LPIPS / DINO / CLIP) live behind protocols and import ``torch`` lazily, so importing
this package without the optional ``[ml]`` extra still succeeds — only constructing a real
backend requires it.
"""

from __future__ import annotations

from viscurate.equivalence.backends import (
    ClipBackend,
    DinoBackend,
    LpipsBackend,
    PerceptualBackend,
    SemanticBackend,
    cosine_distance,
    ssim_distance,
)
from viscurate.equivalence.calibrate import (
    LabeledPair,
    ThresholdFit,
    calibrate_thresholds,
    select_threshold,
)
from viscurate.equivalence.candidates import (
    ENGINEERED_HARD_NEGATIVES,
    candidate_pairs,
    compute_fingerprints,
)
from viscurate.equivalence.compare import BatteryEvaluator, OutputProvider, OutputSet
from viscurate.equivalence.complementary import is_complementary
from viscurate.equivalence.param_alignment import ParamAlignment, load_param_alignment
from viscurate.equivalence.relations import Direction, Relation, RelationResult
from viscurate.equivalence.subsumption import subsumption_search
from viscurate.equivalence.taxonomy import classify

__all__ = [
    "ENGINEERED_HARD_NEGATIVES",
    "BatteryEvaluator",
    "ClipBackend",
    "DinoBackend",
    "Direction",
    "LabeledPair",
    "LpipsBackend",
    "OutputProvider",
    "OutputSet",
    "ParamAlignment",
    "PerceptualBackend",
    "Relation",
    "RelationResult",
    "SemanticBackend",
    "ThresholdFit",
    "calibrate_thresholds",
    "candidate_pairs",
    "classify",
    "compute_fingerprints",
    "cosine_distance",
    "is_complementary",
    "load_param_alignment",
    "select_threshold",
    "ssim_distance",
    "subsumption_search",
]

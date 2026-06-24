"""The equivalence benchmark (CLAUDE.md Phase 4, §3.5.9) — the first go/no-go checkpoint.

This package runs the output-grounded verifier and the text baselines over candidate pairs,
scores both against the designed relation graph ``G0`` (the answer key), and produces the
**divergence table**: the rate at which text judges and the output judge disagree, broken
down by the true (designed) relation, with the engineered hard-negative slice reported
separately. If text and output judges do not diverge, the project premise fails and Phase 5
must not be built (CLAUDE.md Phase 4 / risk register).

Three boundaries are respected:

* the **output-grounded** track is text-blind (it goes through :mod:`viscurate.equivalence`,
  which is handed a :class:`~viscurate.skills.model.ComparatorView`, never a description);
* the **text baselines** live in the separate :mod:`viscurate.baselines` package, which *may*
  read ``description`` (CLAUDE.md §1.2);
* the **answer key** ``G0`` is fixed before any metric runs and is never derived from the
  metrics under test (CLAUDE.md §2.5).
"""

from __future__ import annotations

from viscurate.benchmark.ground_truth import (
    DesignedLabel,
    GroundTruthGraph,
    GroundTruthSpec,
    load_ground_truth,
)
from viscurate.benchmark.human_review import (
    KappaResult,
    ReviewItem,
    cohen_kappa,
    extract_review_slice,
    fleiss_kappa,
    inter_annotator_agreement,
    load_review_labels,
    write_review_template,
)
from viscurate.benchmark.metrics import (
    PRF,
    DivergenceRow,
    SafetyStats,
    abstention_rate,
    confusion_matrix,
    divergence_by_true_relation,
    mergeable_prf,
    per_relation_prf,
    precision_on_distinct,
)
from viscurate.benchmark.report import (
    render_markdown_report,
    try_plot_divergence,
    write_manifest,
    write_report,
)
from viscurate.benchmark.runner import (
    BenchmarkResult,
    CalibrationOutcome,
    PairMeasurement,
    PairOutcome,
    calibrate_from_result,
    labeled_pairs_for_calibration,
    measure_pair,
    run_benchmark,
    split_pairs_by_cluster,
)
from viscurate.benchmark.tracks import Track, Verdict

__all__ = [
    "PRF",
    "BenchmarkResult",
    "CalibrationOutcome",
    "DesignedLabel",
    "DivergenceRow",
    "GroundTruthGraph",
    "GroundTruthSpec",
    "KappaResult",
    "PairMeasurement",
    "PairOutcome",
    "ReviewItem",
    "SafetyStats",
    "Track",
    "Verdict",
    "abstention_rate",
    "calibrate_from_result",
    "cohen_kappa",
    "confusion_matrix",
    "divergence_by_true_relation",
    "extract_review_slice",
    "fleiss_kappa",
    "inter_annotator_agreement",
    "labeled_pairs_for_calibration",
    "load_ground_truth",
    "load_review_labels",
    "measure_pair",
    "mergeable_prf",
    "per_relation_prf",
    "precision_on_distinct",
    "render_markdown_report",
    "run_benchmark",
    "split_pairs_by_cluster",
    "try_plot_divergence",
    "write_manifest",
    "write_report",
    "write_review_template",
]

"""Phase 8 — metrics, baselines, and study aggregation."""

from __future__ import annotations

from viscurate.studies.metrics import (
    AblationDelta,
    AblationResult,
    ActionScore,
    AggregateRow,
    CorrelationResult,
    EquivalenceTrackSummary,
    StudyPoint,
    SummaryStats,
    action_cost,
    aggregate_pareto_front,
    aggregate_points,
    construct_validity,
    equivalence_track_summaries,
    intrinsic_curation_score,
    load_study_points,
    pareto_front,
    score_actions,
    study_point_from_results,
    summarize,
    vision_matters_ablation,
)
from viscurate.studies.report import render_markdown_report, try_plot_pareto, write_study_report

__all__ = [
    "AblationDelta",
    "AblationResult",
    "ActionScore",
    "AggregateRow",
    "CorrelationResult",
    "EquivalenceTrackSummary",
    "StudyPoint",
    "SummaryStats",
    "action_cost",
    "aggregate_pareto_front",
    "aggregate_points",
    "construct_validity",
    "equivalence_track_summaries",
    "intrinsic_curation_score",
    "load_study_points",
    "pareto_front",
    "render_markdown_report",
    "score_actions",
    "study_point_from_results",
    "summarize",
    "try_plot_pareto",
    "vision_matters_ablation",
    "write_study_report",
]

"""Downstream query stream and solver evaluation (CLAUDE.md Phase 7)."""

from __future__ import annotations

from viscurate.downstream.build import (
    DOWNSTREAM_GENERATOR_VERSION,
    QueryBuildConfig,
    build_query_stream,
    load_array,
)
from viscurate.downstream.evaluate import DownstreamResult, QueryScore, run_downstream
from viscurate.downstream.predicates import (
    PredicateResult,
    evaluate_predicate,
    evaluate_predicates,
)
from viscurate.downstream.query import (
    ParamValue,
    PredicateKind,
    PredicateSpec,
    Query,
    QueryManifest,
    QuerySplit,
    QueryStep,
    load_query_manifest,
)
from viscurate.downstream.report import render_markdown_report, write_downstream_report
from viscurate.downstream.solver import (
    ExpectedSkillSolver,
    KeywordRetrievalSolver,
    NoOpSolver,
    PlanExecution,
    SolverAgent,
    SolverPlan,
    execute_plan,
)
from viscurate.downstream.usage import UsageConfig, usage_from_queries

__all__ = [
    "DOWNSTREAM_GENERATOR_VERSION",
    "DownstreamResult",
    "ExpectedSkillSolver",
    "KeywordRetrievalSolver",
    "NoOpSolver",
    "ParamValue",
    "PlanExecution",
    "PredicateKind",
    "PredicateResult",
    "PredicateSpec",
    "Query",
    "QueryBuildConfig",
    "QueryManifest",
    "QueryScore",
    "QuerySplit",
    "QueryStep",
    "SolverAgent",
    "SolverPlan",
    "UsageConfig",
    "build_query_stream",
    "evaluate_predicate",
    "evaluate_predicates",
    "execute_plan",
    "load_array",
    "load_query_manifest",
    "render_markdown_report",
    "run_downstream",
    "usage_from_queries",
    "write_downstream_report",
]

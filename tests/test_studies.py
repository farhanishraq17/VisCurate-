"""Phase 8 — study aggregation, action scoring, Pareto fronts, and reports."""

from __future__ import annotations

import json

import pytest

from viscurate.benchmark import BenchmarkResult, DesignedLabel, Track, Verdict
from viscurate.cli import main
from viscurate.corruption import CorruptionType, IdealAction, IdealActionKind
from viscurate.curation import Action, ActionKind, ActionResult, ActionStatus
from viscurate.equivalence.relations import Relation
from viscurate.studies import (
    StudyPoint,
    aggregate_pareto_front,
    aggregate_points,
    construct_validity,
    equivalence_track_summaries,
    intrinsic_curation_score,
    load_study_points,
    score_actions,
    vision_matters_ablation,
    write_study_report,
)


def _result(kind: ActionKind, status: ActionStatus, primary: str = "", secondary: str = ""):
    return ActionResult(
        action=Action(kind=kind, primary=primary, secondary=secondary),
        status=status,
    )


def _points() -> list[StudyPoint]:
    return [
        StudyPoint(
            method="output-gated",
            gate="output",
            rho=0.1,
            composition="uniform",
            seed=1,
            downstream_success=0.90,
            compression=3,
            action_cost=5,
            intrinsic_score=0.80,
            action_f1=0.80,
        ),
        StudyPoint(
            method="output-gated",
            gate="output",
            rho=0.1,
            composition="uniform",
            seed=2,
            downstream_success=0.80,
            compression=4,
            action_cost=6,
            intrinsic_score=0.70,
            action_f1=0.70,
        ),
        StudyPoint(
            method="text-gated",
            gate="text",
            rho=0.1,
            composition="uniform",
            seed=1,
            downstream_success=0.60,
            compression=5,
            action_cost=4,
            intrinsic_score=0.30,
            action_f1=0.40,
        ),
        StudyPoint(
            method="text-gated",
            gate="text",
            rho=0.1,
            composition="uniform",
            seed=2,
            downstream_success=0.55,
            compression=5,
            action_cost=4,
            intrinsic_score=0.20,
            action_f1=0.30,
        ),
        StudyPoint(
            method="no-curation",
            gate="none",
            rho=0.1,
            composition="uniform",
            seed=1,
            downstream_success=0.50,
            compression=0,
            action_cost=0,
            intrinsic_score=0.0,
        ),
    ]


def test_action_scoring_matches_ideal_key_directionally() -> None:
    ideal = [
        IdealAction(
            kind=IdealActionKind.MERGE,
            type=CorruptionType.DUPLICATE,
            primary="dup_v1",
            secondary="base_v1",
        ),
        IdealAction(
            kind=IdealActionKind.MODIFY,
            type=CorruptionType.METADATA_MISLEAD,
            primary="meta_v1",
        ),
        IdealAction(
            kind=IdealActionKind.REMOVE,
            type=CorruptionType.DEAD_SKILL,
            primary="dead_v1",
        ),
    ]
    log = [
        _result(ActionKind.MERGE, ActionStatus.APPLIED, "dup_v1", "base_v1"),
        _result(ActionKind.MODIFY, ActionStatus.APPLIED, "meta_v1"),
        _result(ActionKind.REMOVE, ActionStatus.REJECTED, "dead_v1"),
        _result(ActionKind.REMOVE, ActionStatus.APPLIED, "unrelated_v1"),
        _result(ActionKind.END, ActionStatus.NOOP),
    ]

    score = score_actions(log, ideal)
    assert (score.tp, score.fp, score.fn) == (2, 1, 1)
    assert score.precision == pytest.approx(2 / 3)
    assert score.recall == pytest.approx(2 / 3)
    assert score.f1 == pytest.approx(2 / 3)
    assert intrinsic_curation_score(log, score) < score.f1  # rejected action spends budget


def test_aggregation_pareto_correlation_and_ablation() -> None:
    points = _points()
    rows = aggregate_points(points)
    output = next(r for r in rows if r.method == "output-gated")
    assert output.n == 2
    assert output.success.mean == pytest.approx(0.85)
    assert output.action_f1 is not None and output.action_f1.mean == pytest.approx(0.75)

    boot_rows = aggregate_points(points, ci_method="bootstrap", bootstrap_samples=100, seed=7)
    boot_output = next(r for r in boot_rows if r.method == "output-gated")
    assert boot_output.success.n == 2
    assert boot_output.success.ci95_low <= boot_output.success.mean <= boot_output.success.ci95_high

    front = aggregate_pareto_front(rows)
    assert any(r.method == "output-gated" for r in front)
    assert any(r.method == "no-curation" for r in front)  # zero action cost is Pareto-relevant

    corr = construct_validity(points)
    assert corr.n == len(points)
    assert corr.pearson is not None and corr.pearson > 0.8
    assert corr.spearman is not None and corr.spearman > 0.8

    ablation = vision_matters_ablation(points)
    assert len(ablation.deltas) == 2
    assert ablation.success_delta.mean == pytest.approx((0.30 + 0.25) / 2)


def test_report_and_point_loaders_emit_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    points = _points()
    paths = write_study_report(points, tmp_path / "out")
    for key in (
        "report",
        "points_csv",
        "aggregates_csv",
        "pareto_csv",
        "ablation_csv",
        "correlations",
        "ablation_summary",
        "manifest",
    ):
        assert paths[key].exists()
    assert "Phase 8 Studies" in paths["report"].read_text()
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["phase"] == 8 and manifest["n_points"] == len(points)

    loaded_csv = load_study_points(paths["points_csv"])
    assert loaded_csv[0].method == points[0].method

    json_path = tmp_path / "points.json"
    json_path.write_text(json.dumps({"points": [p.model_dump(mode="json") for p in points]}))
    loaded_json = load_study_points(json_path)
    assert len(loaded_json) == len(points)


def test_cli_phase8_smoke(tmp_path) -> None:  # type: ignore[no-untyped-def]
    points_path = tmp_path / "points.json"
    points_path.write_text(
        json.dumps({"points": [p.model_dump(mode="json") for p in _points()]}),
        encoding="utf-8",
    )
    out = tmp_path / "phase8"
    assert (
        main(
            [
                "phase8",
                "--points",
                str(points_path),
                "-o",
                str(out),
                "--ci-method",
                "bootstrap",
                "--bootstrap-samples",
                "100",
                "--bootstrap-seed",
                "7",
            ]
        )
        == 0
    )
    assert (out / "report.md").exists()
    assert (out / "manifest.json").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ci_method"] == "bootstrap"


def test_equivalence_track_summaries_reuse_phase4_metrics() -> None:
    pair = ("blur_gaussian_v1", "blur_box_v1")
    truth = {pair: DesignedLabel(Relation.DISTINCT, is_hard_negative=True)}
    result = BenchmarkResult(
        pairs=(pair,),
        truth=truth,
        output_track=Track(
            name="output-grounded",
            kind="output",
            predictions={pair: Verdict(Relation.DISTINCT, mergeable=False)},
        ),
        text_tracks=(
            Track(
                name="embedding-cosine",
                kind="text",
                predictions={pair: Verdict(Relation.EXACT, mergeable=True)},
            ),
            Track(name="llm", kind="text", predictions={}, ran=False, note="not configured"),
        ),
        outcomes=(),
        measurements={},
    )

    rows = {r.track: r for r in equivalence_track_summaries(result)}
    assert rows["output-grounded"].false_merge_rate == pytest.approx(0.0)
    assert rows["embedding-cosine"].false_merge_rate == pytest.approx(1.0)
    assert rows["llm"].ran is False and rows["llm"].mergeable_f1 is None

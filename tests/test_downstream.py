"""Phase 7 — query stream + downstream evaluation."""

from __future__ import annotations

import pytest

from viscurate.config import ThresholdConfig
from viscurate.downstream import (
    ExpectedSkillSolver,
    NoOpSolver,
    PredicateKind,
    QueryBuildConfig,
    build_query_stream,
    run_downstream,
    usage_from_queries,
    write_downstream_report,
)
from viscurate.skills.library import build_builtin_registry
from viscurate.skills.model import Image, Params, Skill


def _build_queries(tmp_path):  # type: ignore[no-untyped-def]
    reg = build_builtin_registry()
    cfg = QueryBuildConfig(seed=17, size=40)
    manifest = build_query_stream(cfg, reg, tmp_path / "queries")
    return reg, manifest, tmp_path / "queries"


def test_query_builder_is_deterministic_and_split_disjoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg = build_builtin_registry()
    cfg = QueryBuildConfig(seed=17, size=40)
    a = build_query_stream(cfg, reg, tmp_path / "a")
    b = build_query_stream(cfg, reg, tmp_path / "b")
    assert a.model_dump(mode="json") == b.model_dump(mode="json")
    assert a.split_counts() == {"dev": 4, "test": 4}
    assert a.splits() == ("dev", "test")
    # A probe manifest with any query input hash would be rejected.
    with pytest.raises(ValueError, match="overlap probe battery"):
        a.assert_disjoint_from_probes([a.entries[0].input_sha256])


def test_usage_stats_are_query_derived(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg, manifest, _ = _build_queries(tmp_path)
    usage = usage_from_queries(manifest, registry_ids=reg.ids())
    assert usage.referenced == manifest.referenced_skill_ids()
    assert usage.usage("grayscale_bt601_v1") > 0
    assert not usage.is_referenced("sepia_tone_v1")  # no default Phase-7 query needs it


def test_expected_solver_succeeds_on_clean_library(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg, manifest, query_dir = _build_queries(tmp_path)
    result = run_downstream(
        manifest,
        query_dir,
        reg,
        ExpectedSkillSolver(),
        thresholds=ThresholdConfig(),
        seed=123,
    )
    assert result.success_rate() == pytest.approx(1.0)
    assert all(s.reference_match and s.predicates_passed for s in result.scores)


def test_noop_solver_fails_reference_matching(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg, manifest, query_dir = _build_queries(tmp_path)
    result = run_downstream(
        manifest,
        query_dir,
        reg,
        NoOpSolver(),
        thresholds=ThresholdConfig(),
        seed=123,
    )
    assert result.success_rate() < 1.0
    assert any(not s.reference_match for s in result.scores)


def test_corrupted_skill_degrades_and_clean_skill_restores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    clean, manifest, query_dir = _build_queries(tmp_path)
    broken = build_builtin_registry()
    invert = broken.get("invert_v1")

    def identity(image: Image, params: Params, seed: int) -> Image:
        return image

    broken.register(
        Skill(
            id=invert.id,
            name=invert.name,
            description=invert.description,
            fn=identity,
            params_schema=invert.params_schema,
            metadata=invert.metadata.model_copy(update={"is_buggy": True}),
        ),
        replace=True,
    )

    noisy = run_downstream(
        manifest,
        query_dir,
        broken,
        ExpectedSkillSolver(),
        thresholds=ThresholdConfig(),
        seed=123,
        splits=("test",),
    )
    restored = run_downstream(
        manifest,
        query_dir,
        clean,
        ExpectedSkillSolver(),
        thresholds=ThresholdConfig(),
        seed=123,
        splits=("test",),
    )
    assert noisy.success_rate("test") < restored.success_rate("test")
    failed = [s for s in noisy.scores if "invert_v1" in s.expected_skill_ids]
    assert failed and not failed[0].success


def test_untrusted_expected_skill_is_blocked(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg, manifest, query_dir = _build_queries(tmp_path)
    invert = reg.get("invert_v1")
    reg.register(
        Skill(
            id=invert.id,
            name=invert.name,
            description=invert.description,
            fn=invert.fn,
            params_schema=invert.params_schema,
            metadata=invert.metadata.model_copy(update={"trusted": False, "provenance": "agent"}),
        ),
        replace=True,
    )
    result = run_downstream(
        manifest,
        query_dir,
        reg,
        ExpectedSkillSolver(),
        thresholds=ThresholdConfig(),
        seed=123,
        splits=("test",),
    )
    blocked = [s for s in result.scores if "invert_v1" in s.expected_skill_ids]
    assert blocked and "blocked untrusted" in blocked[0].error


def test_downstream_report_writes_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg, manifest, query_dir = _build_queries(tmp_path)
    result = run_downstream(
        manifest,
        query_dir,
        reg,
        ExpectedSkillSolver(),
        thresholds=ThresholdConfig(),
        seed=123,
    )
    paths = write_downstream_report(result, tmp_path / "out")
    for key in ("report", "scores_csv", "scores_json", "summary", "manifest"):
        assert paths[key].exists()
    assert "Downstream Evaluation" in paths["report"].read_text()
    assert any(p.kind is PredicateKind.RGBA for q in manifest.entries for p in q.predicates)

"""Phase 9 — experiment manifests, realism audit, and paper artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from viscurate.cli import main
from viscurate.corruption import CorruptionManifest
from viscurate.experiments import Phase9Config, run_phase9
from viscurate.probes.manifest import CC0, ProbeEntry, ProbeManifest
from viscurate.studies import StudyPoint


def _hex(ch: str) -> str:
    return ch * 64


def _write_probe_manifest(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = ProbeManifest(
        generator_version="test",
        canon_version="1.0.0",
        seed=1,
        entries=(
            ProbeEntry(
                probe_id="p0",
                sha256=_hex("a"),
                domain="degenerate",
                channel_format="rgb",
                height=8,
                width=8,
                source="synthetic",
                license=CC0,
                notes="all_black",
            ),
        ),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _write_query_manifest(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "generator_version": "test",
                "canon_version": "1.0.0",
                "seed": 2,
                "entries": [
                    {
                        "query_id": "q0",
                        "split": "test",
                        "instruction": "Invert the image.",
                        "input_id": "q0_input",
                        "input_sha256": _hex("b"),
                        "reference_sha256": _hex("c"),
                        "input_height": 8,
                        "input_width": 8,
                        "reference_height": 8,
                        "reference_width": 8,
                        "pipeline": [{"skill_id": "invert_v1", "params": {}}],
                        "expected_skill_ids": ["invert_v1"],
                        "predicates": [],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_corruption_instance(root: Path) -> None:
    inst = root / "rho010_uniform_seed1_single"
    inst.mkdir(parents=True, exist_ok=True)
    manifest = CorruptionManifest(
        generator_version="test",
        canon_version="1.0.0",
        l0_specs_sha256=_hex("d"),
        g0_sha256=_hex("e"),
        rho=0.1,
        composition="uniform",
        seed=1,
        mode="single",
        n_base=100,
        n_sites=10,
        n_added=1,
        n_skills_lrho=101,
        realized_counts={"duplicate": 1},
    )
    (inst / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    for name in ("corruption_log.json", "library.json", "g_rho.json", "ideal_actions.json"):
        (inst / name).write_text("{}", encoding="utf-8")


def _write_phase8_points(path: Path) -> None:
    point = StudyPoint(
        method="output-gated",
        gate="output",
        rho=0.1,
        composition="uniform",
        seed=1,
        downstream_success=0.75,
        compression=2,
        action_cost=3,
        intrinsic_score=0.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"points": [point.model_dump(mode="json")]}, indent=2),
        encoding="utf-8",
    )


def _config(tmp_path: Path) -> Phase9Config:
    points = tmp_path / "points" / "points.json"
    _write_phase8_points(points)
    _write_probe_manifest(tmp_path / "data" / "probes")
    _write_query_manifest(tmp_path / "data" / "queries")
    _write_corruption_instance(tmp_path / "data" / "corruption")
    return Phase9Config.model_validate(
        {
            "name": "test_run",
            "date": "2026-06-24",
            "seed": 7,
            "paths": {
                "config": "configs/default.yaml",
                "probes_config": "configs/probes.yaml",
                "queries_config": "configs/queries.yaml",
                "corruption_config": "configs/corruption.yaml",
                "ground_truth": "configs/ground_truth_g0.yaml",
                "param_alignment": "configs/param_alignment.yaml",
                "probes_dir": tmp_path / "data" / "probes",
                "oracle_path": tmp_path / "data" / "oracle.json",
                "queries_dir": tmp_path / "data" / "queries",
                "corruption_dir": tmp_path / "data" / "corruption",
                "benchmark_dir": tmp_path / "results" / "phase4",
                "studies_dir": tmp_path / "results" / "phase8",
                "points_path": points,
            },
            "benchmark": {"device": "cpu", "no_ml": True},
        }
    )


def test_phase9_runner_writes_manifest_audit_and_paper_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = _config(tmp_path)
    result = run_phase9(cfg, tmp_path / "phase9", config_path=tmp_path / "phase9.yaml")

    assert result.manifest.exists()
    assert result.audit_json.exists()
    assert result.audit_markdown.exists()
    assert result.repro_script.exists()
    assert result.config_snapshot.exists()
    assert result.paper_artifacts_dir is not None
    assert (result.paper_artifacts_dir / "report.md").exists()
    assert "viscurate phase9" in result.repro_script.read_text(encoding="utf-8")

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["phase"] == 9
    assert manifest["data_manifests"]["probe_battery"]["exists"] is True

    audit = json.loads(result.audit_json.read_text(encoding="utf-8"))
    checks = {c["name"]: c for c in audit["checks"]}
    assert checks["probe_battery"]["status"] == "pass"
    assert checks["query_stream"]["status"] == "pass"
    assert checks["phase4_benchmark"]["status"] == "pending"


def test_cli_phase9_smoke(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = _config(tmp_path)
    cfg_path = tmp_path / "phase9.yaml"
    cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    out = tmp_path / "out"

    assert main(["phase9", "-c", str(cfg_path), "-o", str(out)]) == 0
    assert (out / "run_manifest.json").exists()
    assert (out / "realism_audit.md").exists()

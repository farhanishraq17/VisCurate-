"""Phase-9 experiment runner and paper-artifact bundle writer."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from viscurate.experiments.audit import build_realism_audit, render_audit_markdown
from viscurate.experiments.config import Phase9Config
from viscurate.experiments.manifest import build_run_manifest, repro_commands, write_json
from viscurate.studies import load_study_points, write_study_report

__all__ = ["Phase9Run", "run_phase9"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Phase9Run(_Frozen):
    """Paths written by one Phase-9 run."""

    out_dir: Path
    manifest: Path
    audit_json: Path
    audit_markdown: Path
    repro_script: Path
    config_snapshot: Path
    paper_artifacts_dir: Path | None = None


def _write_config_snapshot(cfg: Phase9Config, out: Path) -> Path:
    path = out / "experiment_config.json"
    path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    return path


def _maybe_write_paper_artifacts(cfg: Phase9Config, out: Path) -> Path | None:
    """Generate Phase-8 tables/figures when a real StudyPoint artifact is configured."""

    points_path = cfg.paths.points_path
    if points_path is None or not points_path.exists():
        return None
    points = load_study_points(points_path)
    if not points:
        return None
    paper_dir = out / "paper_artifacts"
    write_study_report(
        points,
        paper_dir,
        title=f"VisCurate — {cfg.name} Studies",
        output_gate=cfg.output_gate,
        text_gate=cfg.text_gate,
        manifest_extra={
            "phase9_experiment": cfg.name,
            "source_points": str(points_path),
        },
    )
    return paper_dir


def run_phase9(
    cfg: Phase9Config,
    out_dir: str | Path,
    *,
    config_path: str | Path = "configs/phase9.yaml",
) -> Phase9Run:
    """Write the Phase-9 reproducibility bundle without inventing missing results."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config_snapshot = _write_config_snapshot(cfg, out)
    paper_dir = _maybe_write_paper_artifacts(cfg, out)

    manifest = build_run_manifest(cfg)
    if paper_dir is not None:
        manifest["paper_artifacts"] = {
            "path": str(paper_dir),
            "manifest": str(paper_dir / "manifest.json"),
        }
    manifest_path = write_json(out / "run_manifest.json", manifest)

    audit = build_realism_audit(cfg)
    audit_json = out / "realism_audit.json"
    audit_json.write_text(audit.model_dump_json(indent=2), encoding="utf-8")
    audit_md = out / "realism_audit.md"
    audit_md.write_text(render_audit_markdown(audit), encoding="utf-8")

    script = out / "reproduce.sh"
    script.write_text(repro_commands(cfg, config_path=config_path, out_dir=out), encoding="utf-8")

    index = {
        "phase": 9,
        "kind": "artifact_index",
        "manifest": str(manifest_path),
        "audit_json": str(audit_json),
        "audit_markdown": str(audit_md),
        "repro_script": str(script),
        "config_snapshot": str(config_snapshot),
        "paper_artifacts_dir": None if paper_dir is None else str(paper_dir),
    }
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    return Phase9Run(
        out_dir=out,
        manifest=manifest_path,
        audit_json=audit_json,
        audit_markdown=audit_md,
        repro_script=script,
        config_snapshot=config_snapshot,
        paper_artifacts_dir=paper_dir,
    )

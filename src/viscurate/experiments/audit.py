"""Realism and reproducibility audit for Phase-9 paper artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from viscurate.corruption.grid import CorruptionGridConfig
from viscurate.corruption.types import CorruptionManifest
from viscurate.downstream.query import QueryManifest
from viscurate.experiments.config import Phase9Config
from viscurate.probes.manifest import ProbeManifest

__all__ = [
    "AuditCheck",
    "RealismAudit",
    "build_realism_audit",
    "render_audit_markdown",
]

AuditStatus = Literal["pass", "warn", "pending", "fail"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuditCheck(_Frozen):
    """One audit row; ``metadata`` carries machine-readable details for reviewers."""

    name: str
    status: AuditStatus
    detail: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RealismAudit(_Frozen):
    """The full Phase-9 realism audit."""

    checks: tuple[AuditCheck, ...]

    def status_counts(self) -> dict[str, int]:
        counts = Counter(c.status for c in self.checks)
        statuses: tuple[AuditStatus, ...] = ("pass", "warn", "pending", "fail")
        return {k: counts[k] for k in statuses}


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_check(cfg: Phase9Config) -> AuditCheck:
    path = cfg.paths.probes_dir / "manifest.json"
    if not path.exists():
        return AuditCheck(
            name="probe_battery",
            status="pending",
            detail=f"probe manifest not found at {path}",
        )
    try:
        manifest = ProbeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return AuditCheck(name="probe_battery", status="fail", detail=str(exc))
    licenses = sorted({e.license.name for e in manifest.entries})
    return AuditCheck(
        name="probe_battery",
        status="pass",
        detail=f"{len(manifest)} probes, {len(licenses)} concrete license(s)",
        metadata={
            "domain_counts": manifest.domain_counts(),
            "format_counts": manifest.format_counts(),
            "licenses": licenses,
        },
    )


def _query_check(cfg: Phase9Config) -> AuditCheck:
    q_path = cfg.paths.queries_dir / "manifest.json"
    if not q_path.exists():
        return AuditCheck(
            name="query_stream",
            status="pending",
            detail=f"query manifest not found at {q_path}",
        )
    try:
        queries = QueryManifest.model_validate_json(q_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return AuditCheck(name="query_stream", status="fail", detail=str(exc))

    p_path = cfg.paths.probes_dir / "manifest.json"
    disjoint = "not checked"
    status: AuditStatus = "pass"
    if p_path.exists():
        try:
            probes = ProbeManifest.model_validate_json(p_path.read_text(encoding="utf-8"))
            queries.assert_disjoint_from_probes({e.probe_id: e.sha256 for e in probes.entries})
            disjoint = "pass"
        except Exception as exc:
            status = "fail"
            disjoint = str(exc)
    return AuditCheck(
        name="query_stream",
        status=status,
        detail=f"{len(queries)} held-out queries; probe-disjointness: {disjoint}",
        metadata={
            "split_counts": queries.split_counts(),
            "referenced_skills": sorted(queries.referenced_skill_ids()),
        },
    )


def _expected_corruption_count(path: Path) -> int | None:
    if not path.exists():
        return None
    cfg = CorruptionGridConfig.from_yaml(path)
    return len(cfg.rho_values) * len(cfg.compositions) * len(cfg.seeds) * len(cfg.modes)


def _corruption_check(cfg: Phase9Config) -> AuditCheck:
    root = cfg.paths.corruption_dir
    if not root.exists():
        return AuditCheck(
            name="corruption_grid",
            status="pending",
            detail=f"corruption dir not found at {root}",
        )
    manifest_paths = sorted(root.glob("*/manifest.json"))
    if not manifest_paths:
        return AuditCheck(
            name="corruption_grid",
            status="pending",
            detail=f"no corruption instance manifests found under {root}",
        )
    try:
        manifests = [
            CorruptionManifest.model_validate_json(p.read_text(encoding="utf-8"))
            for p in manifest_paths
        ]
    except Exception as exc:
        return AuditCheck(name="corruption_grid", status="fail", detail=str(exc))

    required = ("corruption_log.json", "library.json", "g_rho.json", "ideal_actions.json")
    missing: list[str] = []
    for p in manifest_paths:
        instance = p.parent
        for filename in required:
            if not (instance / filename).exists():
                missing.append(str(instance / filename))
    expected = _expected_corruption_count(cfg.paths.corruption_config)
    status: AuditStatus = "pass"
    if missing:
        status = "fail"
    elif expected is not None and expected != len(manifests):
        status = "warn"
    compositions = sorted({m.composition for m in manifests})
    modes = sorted({m.mode for m in manifests})
    rhos = sorted({m.rho for m in manifests})
    seeds = sorted({m.seed for m in manifests})
    expected_text = "unknown" if expected is None else str(expected)
    return AuditCheck(
        name="corruption_grid",
        status=status,
        detail=(
            f"{len(manifests)}/{expected_text} instance manifest(s); missing files: {len(missing)}"
        ),
        metadata={
            "compositions": compositions,
            "modes": modes,
            "rho_values": rhos,
            "seeds": seeds,
            "missing": missing[:20],
        },
    )


def _artifact_check(
    name: str,
    root: Path,
    required: Sequence[str],
    *,
    calibrated_required: bool = False,
) -> AuditCheck:
    if not root.exists():
        return AuditCheck(name=name, status="pending", detail=f"artifact dir not found at {root}")
    missing = [filename for filename in required if not (root / filename).exists()]
    if missing:
        return AuditCheck(
            name=name,
            status="pending",
            detail=f"missing artifact(s): {', '.join(missing)}",
            metadata={"root": str(root), "missing": missing},
        )
    manifest_path = root / "manifest.json"
    metadata: dict[str, Any] = {"root": str(root)}
    status: AuditStatus = "pass"
    detail = "required artifacts present"
    if manifest_path.exists():
        parsed = _read_json(manifest_path)
        if isinstance(parsed, dict):
            metadata["manifest"] = parsed
            if calibrated_required and parsed.get("thresholds_calibrated") is not True:
                status = "warn"
                detail = "artifacts present, but thresholds are not marked calibrated"
    return AuditCheck(name=name, status=status, detail=detail, metadata=metadata)


def _points_check(cfg: Phase9Config) -> AuditCheck:
    path = cfg.paths.points_path
    if path is None:
        return AuditCheck(
            name="seed_level_points",
            status="pending",
            detail="no Phase-8 StudyPoint file configured",
        )
    if not path.exists():
        return AuditCheck(
            name="seed_level_points",
            status="pending",
            detail=f"configured StudyPoint file not found at {path}",
        )
    n: int | str = "csv"
    if path.suffix.lower() != ".csv":
        raw = _read_json(path)
        data = raw.get("points", raw) if isinstance(raw, dict) else raw
        n = len(data) if isinstance(data, list) else "unknown"
    return AuditCheck(
        name="seed_level_points",
        status="pass",
        detail=f"configured StudyPoint artifact present ({n} rows)",
        metadata={"path": str(path)},
    )


def build_realism_audit(cfg: Phase9Config) -> RealismAudit:
    """Audit whether Phase-9 paper artifacts trace to real manifests."""

    checks = (
        _probe_check(cfg),
        _query_check(cfg),
        _corruption_check(cfg),
        _artifact_check(
            "phase4_benchmark",
            cfg.paths.benchmark_dir,
            ("manifest.json", "divergence.csv", "pairs.csv", "report.md"),
            calibrated_required=True,
        ),
        _points_check(cfg),
        _artifact_check(
            "phase8_studies",
            cfg.paths.studies_dir,
            ("manifest.json", "points.csv", "aggregates.csv", "pareto.csv", "report.md"),
        ),
    )
    return RealismAudit(checks=checks)


def render_audit_markdown(audit: RealismAudit) -> str:
    """Render a concise human-facing audit report."""

    counts = audit.status_counts()
    lines = ["# VisCurate — Phase 9 Realism Audit\n"]
    lines.append(
        "- status counts: " + ", ".join(f"`{k}`={v}" for k, v in counts.items() if v) + "\n"
    )
    lines.append(
        "This audit records whether paper-facing artifacts are backed by manifests. "
        "`pending` means the empirical run is not present; no number is inferred from it.\n"
    )
    for check in audit.checks:
        lines.append(f"## {check.name}\n")
        lines.append(f"- status: **{check.status}**")
        lines.append(f"- detail: {check.detail}")
        if check.metadata:
            compact = json.dumps(check.metadata, sort_keys=True, default=str)
            if len(compact) > 600:
                compact = compact[:600] + "..."
            lines.append(f"- metadata: `{compact}`")
        lines.append("")
    return "\n".join(lines)

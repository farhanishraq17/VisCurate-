"""Phase-9 run manifests and reproducibility command generation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from viscurate import __version__
from viscurate.config import load_config
from viscurate.experiments.config import Phase9Config
from viscurate.skills.canonicalize import CANON_VERSION

__all__ = [
    "build_run_manifest",
    "file_sha256",
    "git_sha",
    "repro_commands",
    "write_json",
]


def file_sha256(path: str | Path) -> str | None:
    """Return the SHA-256 of a file, or ``None`` when the file is absent."""

    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    """Best-effort current git SHA; empty if git is unavailable."""

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return ""
    return out.stdout.decode("ascii", "replace").strip() if out.returncode == 0 else ""


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": file_sha256(path),
    }


def _artifact_manifest(path: Path) -> dict[str, object]:
    manifest_path = path / "manifest.json"
    record = _file_record(manifest_path)
    if manifest_path.exists():
        try:
            parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                record["kind"] = parsed.get("kind")
                record["phase"] = parsed.get("phase")
        except json.JSONDecodeError:
            record["parse_error"] = "invalid json"
    return record


def build_run_manifest(cfg: Phase9Config) -> dict[str, Any]:
    """Build the Phase-9 manifest that ties paper artifacts back to run inputs."""

    runtime_cfg = load_config(cfg.paths.config) if cfg.paths.config.exists() else load_config(None)
    paths = cfg.paths
    return {
        "phase": 9,
        "kind": "experiment_runner",
        "name": cfg.name,
        "date": cfg.date,
        "seed": cfg.seed,
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "viscurate_version": __version__,
        "canon_version": CANON_VERSION,
        "thresholds": runtime_cfg.thresholds.model_dump(mode="json"),
        "benchmark": cfg.benchmark.model_dump(mode="json"),
        "model_versions": {
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "lpips": _package_version("lpips"),
            "timm": _package_version("timm"),
            "open_clip_torch": _package_version("open-clip-torch"),
            "scikit_image": _package_version("scikit-image"),
        },
        "config_files": {
            "default": _file_record(paths.config),
            "probes": _file_record(paths.probes_config),
            "queries": _file_record(paths.queries_config),
            "corruption": _file_record(paths.corruption_config),
            "ground_truth": _file_record(paths.ground_truth),
            "param_alignment": _file_record(paths.param_alignment),
        },
        "data_manifests": {
            "probe_battery": _file_record(paths.probes_dir / "manifest.json"),
            "oracle": _file_record(paths.oracle_path),
            "query_stream": _file_record(paths.queries_dir / "manifest.json"),
        },
        "result_manifests": {
            "phase4_benchmark": _artifact_manifest(paths.benchmark_dir),
            "phase8_studies": _artifact_manifest(paths.studies_dir),
            "phase8_points": _file_record(paths.points_path) if paths.points_path else None,
        },
    }


def repro_commands(cfg: Phase9Config, *, config_path: str | Path, out_dir: str | Path) -> str:
    """Return a bash script documenting the one-command reproduction flow."""

    p = cfg.paths
    bench_flags = [
        "viscurate run-benchmark",
        f"-c {p.config}",
        f"--probes-dir {p.probes_dir}",
        f"--ground-truth {p.ground_truth}",
        f"--param-alignment {p.param_alignment}",
        f"-o {p.benchmark_dir}",
        f"--device {cfg.benchmark.device}",
        f"-k {cfg.benchmark.k}",
        f"--screening {cfg.benchmark.screening}",
    ]
    if cfg.benchmark.no_ml:
        bench_flags.append("--no-ml")
    if cfg.benchmark.clip:
        bench_flags.append("--clip")
    if cfg.benchmark.calibrate:
        bench_flags.append("--calibrate")
        if cfg.date:
            bench_flags.append(f"--date {cfg.date}")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'python -m pip install -e ".[dev,ml,viz]"',
        f"viscurate build-probes -c {p.probes_config} -o {p.probes_dir}",
        f"viscurate freeze-oracle --probes-dir {p.probes_dir} -o {p.oracle_path} --seed {cfg.seed}",
        " ".join(bench_flags),
        f"viscurate corrupt -c {p.corruption_config} --ground-truth {p.ground_truth} "
        f"-o {p.corruption_dir} --probes-dir {p.probes_dir}",
        f"viscurate build-queries -c {p.queries_config} -o {p.queries_dir} "
        f"--probes-dir {p.probes_dir}",
    ]
    if p.points_path is not None:
        lines.append(
            f"viscurate phase8 --points {p.points_path} -o {p.studies_dir} "
            "--title 'VisCurate - Phase 8 Studies'"
        )
    else:
        lines.append(
            "# Add real seed-level StudyPoint rows, then run: "
            "viscurate phase8 --points <points.json|points.csv>"
        )
    lines.append(f"viscurate phase9 -c {config_path} -o {out_dir}")
    return "\n".join(lines) + "\n"


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return p

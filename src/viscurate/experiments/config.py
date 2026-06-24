"""Phase-9 experiment configuration.

This config does not hide experiment execution inside opaque code. It records the paths and
knobs needed to reproduce the lower-phase commands, then the Phase-9 runner fingerprints and
audits the resulting artifacts.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BenchmarkRunConfig",
    "ExperimentPaths",
    "Phase9Config",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkRunConfig(_Frozen):
    """The Phase-4 benchmark knobs that must be recorded for reproducibility."""

    device: str = "cpu"
    no_ml: bool = True
    clip: bool = False
    calibrate: bool = False
    screening: int = Field(default=12, gt=0)
    k: int = Field(default=5, gt=0)


class ExperimentPaths(_Frozen):
    """All paths Phase 9 fingerprints or uses to generate paper-facing artifacts."""

    config: Path = Path("configs/default.yaml")
    probes_config: Path = Path("configs/probes.yaml")
    queries_config: Path = Path("configs/queries.yaml")
    corruption_config: Path = Path("configs/corruption.yaml")
    ground_truth: Path = Path("configs/ground_truth_g0.yaml")
    param_alignment: Path = Path("configs/param_alignment.yaml")

    probes_dir: Path = Path("data/probe_images")
    oracle_path: Path = Path("data/oracle/oracle.json")
    queries_dir: Path = Path("data/queries")
    corruption_dir: Path = Path("data/corruption")

    benchmark_dir: Path = Path("results/phase4_benchmark")
    studies_dir: Path = Path("results/phase8_studies")
    points_path: Path | None = None


class Phase9Config(_Frozen):
    """One declarative Phase-9 experiment bundle."""

    name: str = "cvpr_pilot"
    date: str = ""
    seed: int = 1234
    paths: ExperimentPaths = ExperimentPaths()
    benchmark: BenchmarkRunConfig = BenchmarkRunConfig()
    output_gate: str = "output"
    text_gate: str = "text"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Phase9Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

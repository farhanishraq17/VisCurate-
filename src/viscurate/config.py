"""Pydantic-validated configuration loaded from YAML.

Two project rules are enforced here:

* **No magic literals in code.** Thresholds (ε, τ_perceptual, τ_semantic, the abstention
  band δ), the executor timeout, and seeds all come from config, recorded with the run
  (CLAUDE.md §3.5.5, §5).
* **Calibrated values are not guessed.** :class:`ThresholdConfig` defaults are explicit
  *placeholders* and carry ``calibrated=False`` plus provenance fields; any reported
  metric must use values calibrated on the human-labeled validation split and stamped
  with the split hash and date.

Models are frozen and forbid unknown keys, so a typo in a YAML file is an error, not a
silently ignored field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Config",
    "CurationConfig",
    "DownstreamConfig",
    "ExecutorConfig",
    "LoggingConfig",
    "PathsConfig",
    "RunConfig",
    "ThresholdConfig",
    "load_config",
]


class _Strict(BaseModel):
    """Base: frozen + reject unknown keys so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(_Strict):
    name: str = "default"
    # Single root seed; every component derives its own via viscurate.rng.SeedManager.
    seed: int = 1234


class PathsConfig(_Strict):
    data_dir: Path = Path("data")
    probe_dir: Path = Path("data/probe_images")
    query_dir: Path = Path("data/queries")
    oracle_dir: Path = Path("data/oracle")
    results_dir: Path = Path("results")
    configs_dir: Path = Path("configs")


class ExecutorConfig(_Strict):
    """Lightweight isolation for the trusted starter set (CLAUDE.md D6 / §3.2).

    ``allow_untrusted`` MUST stay False until the Phase-6 hardened sandbox is reviewed:
    ``trusted=False`` skills are blocked from execution.
    """

    timeout_s: float = Field(default=10.0, gt=0)
    max_memory_mb: int | None = Field(default=2048, gt=0)
    allow_untrusted: bool = False


class LoggingConfig(_Strict):
    level: str = "INFO"
    # `json_format` rather than `json`: a field named `json` shadows a BaseModel attr.
    json_format: bool = True


class ThresholdConfig(_Strict):
    """Operating point for the output-grounded verifier (Phase 3).

    The numeric defaults are PLACEHOLDERS for wiring/tests, never authoritative. A run
    that reports equivalence metrics must set ``calibrated=True`` and fill the provenance
    fields (CLAUDE.md §3.5.5).
    """

    exact_epsilon: float = Field(default=1.0 / 255.0, ge=0)  # L∞ in [0,1], rounding only
    perceptual_lpips: float = Field(default=0.05, ge=0)  # worst-case LPIPS for PERCEPTUAL
    perceptual_ssim: float = Field(default=0.10, ge=0)  # max (1−SSIM) structural floor
    semantic_dino: float = Field(default=0.15, ge=0)  # p90 DINO cosine dist for SEMANTIC
    semantic_quantile: float = Field(default=0.90, ge=0, le=1)  # aggregation quantile (p90)
    complementary_lpips: float = Field(default=0.05, ge=0)  # commutation tolerance
    abstention_delta: float = Field(default=0.10, ge=0, le=1)  # band half-width around τ
    calibrated: bool = False
    calibration_split_hash: str | None = None
    calibration_date: str | None = None

    @model_validator(mode="after")
    def _provenance_required_when_calibrated(self) -> ThresholdConfig:
        if self.calibrated and not (self.calibration_split_hash and self.calibration_date):
            raise ValueError("calibrated=True requires calibration_split_hash and calibration_date")
        return self


class CurationConfig(_Strict):
    """The curation episode budget + usage gate (CLAUDE.md Phase 6, roadmap open item 5).

    ``budget`` caps the agent's actions per episode — the action-cost axis of the Pareto front
    (CLAUDE.md §3.4). ``usage_fold_threshold`` is the usage at/above which folding a skill away
    (parameterize) is flagged as losing a used skill (CLAUDE.md §3.5.7).
    """

    budget: int = Field(default=200, gt=0)
    usage_fold_threshold: int = Field(default=1, ge=0)


class DownstreamConfig(_Strict):
    """Phase-7 query generation + query-derived usage knobs."""

    query_seed: int = 2026
    query_size: int = Field(default=96, gt=8)
    dev_repeats: int = Field(default=1, ge=1)
    test_repeats: int = Field(default=1, ge=1)
    usage_base_count: int = Field(default=20, ge=1)
    usage_zipf_alpha: float = Field(default=1.2, ge=0)


class Config(_Strict):
    run: RunConfig = RunConfig()
    paths: PathsConfig = PathsConfig()
    executor: ExecutorConfig = ExecutorConfig()
    logging: LoggingConfig = LoggingConfig()
    thresholds: ThresholdConfig = ThresholdConfig()
    curation: CurationConfig = CurationConfig()
    downstream: DownstreamConfig = DownstreamConfig()

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> Config:
        return cls.model_validate(data or {})


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a YAML config. With no path, return validated defaults."""
    if path is None:
        return Config()
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is not None and not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping, got {type(raw).__name__}: {p}")
    return Config.from_mapping(raw)

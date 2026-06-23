from __future__ import annotations

import pytest
from pydantic import ValidationError

from viscurate.config import Config, ThresholdConfig, load_config


def test_defaults_load_without_file() -> None:
    cfg = load_config(None)
    assert cfg.run.seed == 1234
    assert cfg.executor.allow_untrusted is False  # hard safety default


def test_loads_yaml(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("run:\n  name: exp1\n  seed: 7\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.run.name == "exp1"
    assert cfg.run.seed == 7


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.from_mapping({"run": {"nope": 1}})


def test_config_is_frozen() -> None:
    cfg = Config()
    with pytest.raises(ValidationError):
        cfg.run.seed = 99  # type: ignore[misc]


def test_calibrated_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        ThresholdConfig(calibrated=True)
    ok = ThresholdConfig(
        calibrated=True, calibration_split_hash="abc123", calibration_date="2026-06-23"
    )
    assert ok.calibrated


def test_repo_default_config_is_valid() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg.thresholds.calibrated is False  # placeholders, not authoritative
    assert cfg.thresholds.exact_epsilon == pytest.approx(1.0 / 255.0)

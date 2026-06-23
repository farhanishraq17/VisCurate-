from __future__ import annotations

import numpy as np

from viscurate.config import ExecutorConfig
from viscurate.skills._testkit import make_crash_skill, make_sleep_skill
from viscurate.skills.executor import SandboxedExecutor
from viscurate.skills.library import build_builtin_registry
from viscurate.skills.model import Image


def test_untrusted_skill_is_blocked(probe_rgb: Image) -> None:
    ex = SandboxedExecutor(ExecutorConfig(allow_untrusted=False))
    result = ex.run(make_sleep_skill(trusted=False), probe_rgb)
    assert result.blocked is True
    assert result.ok is False
    assert result.output is None


def test_timeout_returns_error_not_hang(probe_rgb: Image) -> None:
    ex = SandboxedExecutor(ExecutorConfig(timeout_s=2.0))
    result = ex.run(make_sleep_skill(), probe_rgb, {"seconds": 30.0})
    assert result.timed_out is True
    assert result.ok is False
    assert result.duration_s < 10.0  # actually returned, did not hang


def test_crashing_skill_returns_structured_error(probe_rgb: Image) -> None:
    ex = SandboxedExecutor(ExecutorConfig(timeout_s=15.0))
    result = ex.run(make_crash_skill(), probe_rgb)
    assert result.ok is False
    assert result.error is not None
    assert "RuntimeError" in result.error


def test_successful_execution_matches_in_process(probe_rgb: Image) -> None:
    ex = SandboxedExecutor(ExecutorConfig(timeout_s=30.0))
    skill = build_builtin_registry().get("blur_gaussian_v1")
    result = ex.run(skill, probe_rgb, {"ksize": 5, "sigma": 0.0}, seed=0)
    assert result.ok is True
    assert result.output is not None
    in_process = skill.run(probe_rgb, {"ksize": 5, "sigma": 0.0}, seed=0)
    assert np.array_equal(result.output, in_process)


def test_sandboxed_runs_are_byte_identical(probe_rgb: Image) -> None:
    ex = SandboxedExecutor(ExecutorConfig(timeout_s=30.0))
    skill = build_builtin_registry().get("rotate_90_v1")
    r1 = ex.run(skill, probe_rgb)
    r2 = ex.run(skill, probe_rgb)
    assert r1.ok and r2.ok
    assert r1.output is not None and r2.output is not None
    assert np.array_equal(r1.output, r2.output)

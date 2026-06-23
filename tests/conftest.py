"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from viscurate.skills.library import build_builtin_registry
from viscurate.skills.model import Image
from viscurate.skills.registry import SkillRegistry


@pytest.fixture
def probe_rgb() -> Image:
    """A deterministic 64x80 RGB probe (fixed seed → byte-stable)."""
    rng = np.random.default_rng(20260623)
    return rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)


@pytest.fixture
def registry() -> SkillRegistry:
    return build_builtin_registry()

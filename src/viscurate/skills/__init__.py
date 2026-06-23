"""The skill harness: model, registry, canonicalization, sandboxed executor, library.

A *skill* is a deterministic ``fn(image, params, seed) -> image``. This package is the
project's executable substrate and is deliberately ML-dependency-free (CLAUDE.md §4).
"""

from __future__ import annotations

import cv2

from viscurate.skills.model import (
    ParamSpec,
    ParamsSchema,
    Skill,
    SkillMetadata,
    SkillSpec,
)
from viscurate.skills.registry import SkillRegistry

# Reproducibility: pin OpenCV to a single thread so skill outputs are deterministic across
# runs and platforms (CLAUDE.md §1.4). Importing any skill module triggers this package
# __init__, so every execution path (registry, oracle, subprocess worker) is covered.
cv2.setNumThreads(1)

__all__ = [
    "ParamSpec",
    "ParamsSchema",
    "Skill",
    "SkillMetadata",
    "SkillRegistry",
    "SkillSpec",
]

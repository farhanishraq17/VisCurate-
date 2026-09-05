"""Regression: a pair with no comparable output must never be certified mergeable.

The perceptual stage aggregates by worst case over the sweep × probes. Seeding that running
maximum at ``-1.0`` and returning it unguarded means an EMPTY comparison reports a worst-case
LPIPS of ``-1.0``, which sits below every perceptual threshold — so two skills that never once
produced a comparable output were certified PERCEPTUAL, i.e. licensed for merge on no evidence.
A skill that raises on every probe (a dead skill, or a real library function that cannot run on
this battery) reaches that path.

The pixel stage already guarded this with ``saw_common``; these tests pin the same contract on
the perceptual stage, and only trigger when a perceptual backend is attached — which is why the
no-backend test suite never caught it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from viscurate.config import ThresholdConfig
from viscurate.equivalence.compare import BatteryEvaluator
from viscurate.equivalence.relations import Relation
from viscurate.equivalence.taxonomy import classify
from viscurate.skills.model import Image, Params, ParamsSchema, Skill, SkillMetadata


class ZeroDistanceBackend:
    """A perceptual backend that always reports "identical" — the most permissive case.

    Using it makes the test independent of LPIPS weights: if the guard is missing, the empty
    comparison passes; if the guard is present, the pair falls through regardless of the backend.
    """

    name = "zero-distance-stub"

    def distance(self, a: Any, b: Any) -> float:
        return 0.0

    def features(self, images: Any) -> Any:
        return np.zeros((len(list(images)), 4), dtype=np.float32)


def _skill(sid: str, fn: Any) -> Skill:
    return Skill(
        id=sid,
        name=sid,
        description="",
        fn=fn,
        params_schema=ParamsSchema(params=()),
        metadata=SkillMetadata(family="t", trusted=True),
    )


def _identity(image: Image, params: Params, seed: int) -> Image:
    return image.copy()


def _always_raises(image: Image, params: Params, seed: int) -> Image:
    raise ValueError("this skill never works")


@pytest.fixture
def setup() -> tuple[BatteryEvaluator, dict[str, Skill], ThresholdConfig]:
    battery = [(f"p{i}", np.full((16, 16, 3), (i * 31) % 256, np.uint8)) for i in range(6)]
    skills = {"good": _skill("good", _identity), "broken": _skill("broken", _always_raises)}
    provider = BatteryEvaluator(list(skills.values()), battery, seed=0)
    return provider, skills, ThresholdConfig()


def test_broken_skill_produces_no_outputs(setup: Any) -> None:
    provider, _skills, _thr = setup
    out = provider.outputs("broken")
    assert out.probe_ids == ()
    assert len(out.errors) == 6


@pytest.mark.parametrize(("a", "b"), [("broken", "broken"), ("good", "broken")])
def test_empty_common_probe_set_is_not_mergeable(setup: Any, a: str, b: str) -> None:
    provider, skills, thr = setup
    result = classify(
        skills[a].comparator_view(),
        skills[b].comparator_view(),
        provider,
        thresholds=thr,
        perceptual=ZeroDistanceBackend(),
    )
    assert not result.licenses_merge, f"{a} vs {b} was licensed for merge on zero evidence"
    assert result.relation is not Relation.PERCEPTUAL
    assert result.relation is Relation.DISTINCT


def test_identical_skills_still_certify_exact(setup: Any) -> None:
    """The guard must not suppress a genuine merge: a self-pair that DOES run stays EXACT."""
    provider, skills, thr = setup
    result = classify(
        skills["good"].comparator_view(),
        skills["good"].comparator_view(),
        provider,
        thresholds=thr,
        perceptual=ZeroDistanceBackend(),
    )
    assert result.relation is Relation.EXACT
    assert result.licenses_merge

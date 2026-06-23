from __future__ import annotations

import numpy as np
import pytest

from viscurate.skills.library import build_builtin_registry, builtin_fn_resolver
from viscurate.skills.model import Image
from viscurate.skills.registry import SkillRegistry


def test_duplicate_registration_rejected(registry: SkillRegistry) -> None:
    skill = registry.all()[0]
    with pytest.raises(ValueError):
        registry.register(skill)


def test_json_round_trip_preserves_specs(registry: SkillRegistry) -> None:
    text = registry.to_json()
    reloaded = SkillRegistry.from_json(text, builtin_fn_resolver)
    assert reloaded.ids() == registry.ids()
    for original in registry:
        assert reloaded.get(original.id).to_spec() == original.to_spec()


def test_reloaded_skill_executes_identically(probe_rgb: Image) -> None:
    original = build_builtin_registry()
    reloaded = SkillRegistry.from_json(original.to_json(), builtin_fn_resolver)
    for sid in ("blur_gaussian_v1", "rotate_90_v1", "grayscale_bt601_v1"):
        a = original.get(sid).run(probe_rgb, seed=3)
        b = reloaded.get(sid).run(probe_rgb, seed=3)
        assert np.array_equal(a, b)


def test_save_and_load(tmp_path, registry: SkillRegistry) -> None:
    path = tmp_path / "registry.json"
    registry.save(path)
    reloaded = SkillRegistry.load(path, builtin_fn_resolver)
    assert len(reloaded) == len(registry)


def test_unknown_schema_version_rejected() -> None:
    with pytest.raises(ValueError):
        SkillRegistry.from_dict({"schema_version": "999", "skills": []}, builtin_fn_resolver)

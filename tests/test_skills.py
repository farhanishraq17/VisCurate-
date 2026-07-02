from __future__ import annotations

import numpy as np
import pytest

from viscurate.skills.canonicalize import canonicalize, max_abs_pixel_diff
from viscurate.skills.library import build_builtin_registry
from viscurate.skills.model import Image

_REGISTRY = build_builtin_registry()
_SKILL_IDS = _REGISTRY.ids()


def test_expected_skill_count() -> None:
    assert len(_SKILL_IDS) == 100
    assert len(set(_SKILL_IDS)) == len(_SKILL_IDS)  # ids unique


def test_determinism_flags_assigned() -> None:
    seeded = {s.id for s in _REGISTRY if s.metadata.seeded_stochastic}
    precision = {s.id for s in _REGISTRY if s.metadata.precision_sensitive}
    platform = {s.id for s in _REGISTRY if s.metadata.platform_sensitive}
    assert {
        "add_gaussian_noise_v1",
        "add_salt_pepper_noise_v1",
        "random_crop_v1",
        "value_noise_synthesize_v1",
    } <= seeded
    assert precision == {"low_pass_fft_v1", "high_pass_fft_v1"}
    assert platform == {"palette_reduce_kmeans_v1"}


@pytest.mark.parametrize("skill_id", _SKILL_IDS)
def test_skill_runs_deterministically_and_canonicalizes(skill_id: str, probe_rgb: Image) -> None:
    skill = _REGISTRY.get(skill_id)
    out1 = skill.run(probe_rgb, seed=11)
    out2 = skill.run(probe_rgb, seed=11)
    assert isinstance(out1, np.ndarray)
    assert np.array_equal(out1, out2), f"{skill_id} is non-deterministic"
    canon = canonicalize(out1)  # must not raise
    assert canon.rgb.shape[2] == 3


# --- planted relations (sanity checks for the later equivalence benchmark) -------------


def test_rotate90_subsumed_by_canvas_rotation(probe_rgb: Image) -> None:
    r90 = _REGISTRY.get("rotate_90_v1").run(probe_rgb)
    canvas = _REGISTRY.get("rotate_canvas_degrees_v1").run(probe_rgb, {"degrees": 90.0})
    assert np.array_equal(r90, canvas)
    canvas180 = _REGISTRY.get("rotate_canvas_degrees_v1").run(probe_rgb, {"degrees": 180.0})
    assert np.array_equal(_REGISTRY.get("rotate_180_v1").run(probe_rgb), canvas180)


def test_rotate45_equals_canvas_rotation_at_45(probe_rgb: Image) -> None:
    r45 = _REGISTRY.get("rotate_45_v1").run(probe_rgb)
    canvas = _REGISTRY.get("rotate_canvas_degrees_v1").run(probe_rgb, {"degrees": 45.0})
    assert np.array_equal(r45, canvas)  # fixed-angle specialization ⊑ general rotation


def test_mask_to_rgba_emits_alpha(probe_rgb: Image) -> None:
    out = _REGISTRY.get("mask_to_rgba_v1").run(probe_rgb, {"thresh": 128})
    assert out.ndim == 3 and out.shape[2] == 4  # RGBA
    canon = canonicalize(out)
    assert canon.alpha is not None  # alpha tracked separately by the contract


def test_center_crop_subsumed_by_bounding_box(probe_rgb: Image) -> None:
    center = _REGISTRY.get("crop_center_percentage_v1").run(probe_rgb, {"percent": 50.0})
    bbox = _REGISTRY.get("crop_bounding_box_v1").run(
        probe_rgb, {"left": 0.25, "top": 0.25, "width": 0.5, "height": 0.5}
    )
    assert np.array_equal(center, bbox)


def test_center_crop_matches_centered_bbox_on_odd_size() -> None:
    image = np.arange(65 * 67 * 3, dtype=np.uint8).reshape(65, 67, 3)
    center = _REGISTRY.get("crop_center_percentage_v1").run(image, {"percent": 50.0})
    bbox = _REGISTRY.get("crop_bounding_box_v1").run(
        image, {"left": 0.25, "top": 0.25, "width": 0.5, "height": 0.5}
    )
    assert np.array_equal(center, bbox)


def test_linear_stretch_is_percentile_0_100(probe_rgb: Image) -> None:
    linear = _REGISTRY.get("linear_contrast_stretch_v1").run(probe_rgb)
    pct = _REGISTRY.get("contrast_stretching_percentile_v1").run(
        probe_rgb, {"low_pct": 0.0, "high_pct": 100.0}
    )
    assert np.array_equal(linear, pct)


def test_gaussian_and_box_blur_are_distinct_at_large_k(probe_rgb: Image) -> None:
    g = _REGISTRY.get("blur_gaussian_v1").run(probe_rgb, {"ksize": 9, "sigma": 0.0})
    b = _REGISTRY.get("blur_box_v1").run(probe_rgb, {"ksize": 9})
    diff = max_abs_pixel_diff(canonicalize(g), canonicalize(b))
    assert diff > 1.0 / 255.0  # clearly not EXACT -> DISTINCT hard negative


def test_seeded_noise_matches_at_same_seed_differs_across_seeds(probe_rgb: Image) -> None:
    noise = _REGISTRY.get("add_gaussian_noise_v1")
    a = noise.run(probe_rgb, {"sigma": 20.0}, seed=1)
    a2 = noise.run(probe_rgb, {"sigma": 20.0}, seed=1)
    b = noise.run(probe_rgb, {"sigma": 20.0}, seed=2)
    assert np.array_equal(a, a2)
    assert not np.array_equal(a, b)


def test_degenerate_inputs_do_not_crash_simple_skills() -> None:
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    for sid in ("flip_horizontal_v1", "invert_v1", "brightness_add_v1", "grayscale_bt601_v1"):
        out = _REGISTRY.get(sid).run(black)
        assert isinstance(out, np.ndarray)

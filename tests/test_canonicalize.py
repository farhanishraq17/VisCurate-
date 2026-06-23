from __future__ import annotations

import numpy as np

from viscurate.skills.canonicalize import (
    canonicalize,
    content_hash,
    mask_iou,
    max_abs_pixel_diff,
)


def test_float_range_and_uint8_copy() -> None:
    img = np.full((4, 4, 3), 255, dtype=np.uint8)
    c = canonicalize(img)
    assert c.rgb.dtype == np.float32
    assert c.rgb.max() <= 1.0 and c.rgb.min() >= 0.0
    assert c.rgb_u8.dtype == np.uint8
    assert c.rgb.shape == (4, 4, 3)


def test_grayscale_replicated_to_three_channels() -> None:
    gray = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    c = canonicalize(gray)
    assert c.rgb.shape == (2, 2, 3)
    # All three channels equal for a replicated grayscale image.
    assert np.allclose(c.rgb[:, :, 0], c.rgb[:, :, 1])
    assert np.allclose(c.rgb[:, :, 1], c.rgb[:, :, 2])


def test_rgba_composited_over_midgray_and_alpha_tracked() -> None:
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 0] = 255  # pure red
    rgba[..., 3] = 0  # fully transparent -> should composite to mid-gray
    c = canonicalize(rgba)
    assert c.alpha is not None
    assert np.allclose(c.rgb, 0.5, atol=1e-6)  # transparent over mid-gray


def test_binary_mask_detected_and_iou() -> None:
    a = np.zeros((4, 4), dtype=np.uint8)
    a[:2] = 255
    b = np.zeros((4, 4), dtype=np.uint8)
    b[:3] = 255
    ca, cb = canonicalize(a), canonicalize(b)
    assert ca.is_binary_mask and cb.is_binary_mask
    # intersection = 2 rows, union = 3 rows -> IoU = 2/3
    assert mask_iou(ca, cb) == (2 * 4) / (3 * 4)
    assert mask_iou(ca, ca) == 1.0


def test_shape_gate_returns_inf() -> None:
    a = canonicalize(np.zeros((4, 4, 3), dtype=np.uint8))
    b = canonicalize(np.zeros((4, 5, 3), dtype=np.uint8))
    assert max_abs_pixel_diff(a, b) == float("inf")


def test_content_hash_collides_for_identical_differs_for_shape() -> None:
    a = canonicalize(np.full((4, 4, 3), 10, dtype=np.uint8))
    a2 = canonicalize(np.full((4, 4, 3), 10, dtype=np.uint8))
    b = canonicalize(np.full((4, 5, 3), 10, dtype=np.uint8))
    assert content_hash(a) == content_hash(a2)
    assert content_hash(a) != content_hash(b)


def test_max_abs_pixel_diff_zero_for_equal() -> None:
    img = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    c = canonicalize(img)
    assert max_abs_pixel_diff(c, c) == 0.0

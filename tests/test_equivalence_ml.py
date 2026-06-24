"""Phase 3 — real backend smoke test (gated on the ``[ml]`` extra).

These load actual model weights (LPIPS AlexNet, DINO ViT-B/16, CLIP ViT-B/32), so they are
marked ``slow`` and skipped if the extra is not installed. They confirm the backends produce
sane distances (identical inputs → ~0, different inputs → larger) and that models load **one at
a time** (each backend is closed before the next), the 6 GB-budget discipline (CLAUDE.md D5).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _img(seed: int, side: int = 48) -> np.ndarray:
    return np.random.default_rng(seed).random((side, side, 3)).astype(np.float32)


def test_lpips_identical_is_zero_and_monotone() -> None:
    pytest.importorskip("lpips")
    from viscurate.equivalence import LpipsBackend

    a = _img(0)
    b = np.clip(a + 0.5, 0.0, 1.0).astype(np.float32)
    with LpipsBackend() as be:
        d_same = be.distance(a, a)
        d_diff = be.distance(a, b)
    assert d_same == pytest.approx(0.0, abs=1e-4)
    assert d_diff > d_same


def test_dino_features_and_identical_cosine_zero() -> None:
    pytest.importorskip("timm")
    from viscurate.equivalence import DinoBackend, cosine_distance

    a, b = _img(1), _img(2)
    with DinoBackend() as be:
        feats = be.features([a, a, b])
    assert feats.shape[0] == 3
    assert feats.shape[1] >= 256  # ViT-B/16 feature dim (768)
    assert cosine_distance(feats[0], feats[1]) == pytest.approx(0.0, abs=1e-5)
    assert cosine_distance(feats[0], feats[2]) > 1e-4


def test_clip_optional_view() -> None:
    pytest.importorskip("open_clip")
    from viscurate.equivalence import ClipBackend, cosine_distance

    a = _img(3, side=40)
    with ClipBackend() as be:
        feats = be.features([a, a])
    assert feats.shape[0] == 2
    assert cosine_distance(feats[0], feats[1]) == pytest.approx(0.0, abs=1e-5)


def test_ssim_distance_identical_is_zero() -> None:
    pytest.importorskip("skimage")
    from viscurate.equivalence import ssim_distance

    a = _img(4, side=32)
    assert ssim_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert ssim_distance(a, np.zeros_like(a)) > 0.0
    assert ssim_distance(a, a[:16]) == float("inf")  # shape gate

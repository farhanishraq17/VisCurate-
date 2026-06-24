"""The output canonicalization contract (CLAUDE.md §1.3).

Skills emit heterogeneous outputs — grayscale, RGBA, binary masks, edge maps,
shape-changing geometric ops. Pixel/LPIPS comparison needs a *fixed, versioned* rule so
that relation labels are reproducible. This module is that rule. It is consumed only by
the verifier, which sees outputs and never text.

Rules:

* **dtype/range** — coerce to ``float32 ∈ [0,1]`` for metrics; keep a ``uint8`` copy for
  hashing. ``uint8`` ÷255, ``uint16`` ÷65535, ``bool`` → {0,1}, float clipped to [0,1].
* **channels** — 1-channel → replicate to 3; RGBA → composite over a fixed mid-gray
  background for the RGB view *and* keep alpha separately (alpha-IoU / alpha-L∞).
* **shape** — identical shape is a precondition for EXACT/PERCEPTUAL; ``max_abs_pixel_diff``
  returns ``inf`` on a shape mismatch (the §3.5.1 shape gate).
* **binary masks** — single-channel {0,max} outputs are flagged and compared by exact
  match + IoU, not LPIPS.

``CANON_VERSION`` is part of the contract; bump it on any behavioural change and store it
alongside probe manifests so labels remain reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "CANON_VERSION",
    "MIDGRAY",
    "Canonical",
    "alpha_linf",
    "canonicalize",
    "content_hash",
    "mask_iou",
    "max_abs_pixel_diff",
    "same_shape",
]

CANON_VERSION = "1.0.0"
MIDGRAY = 0.5  # fixed background for RGBA compositing

NDArrayF = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Canonical:
    """A canonicalized output: a float32 RGB view, a uint8 hash view, optional alpha/mask."""

    version: str
    height: int
    width: int
    n_input_channels: int
    is_binary_mask: bool
    rgb: NDArrayF  # (H, W, 3) float32 in [0, 1], RGBA composited over mid-gray
    rgb_u8: npt.NDArray[np.uint8]  # (H, W, 3) uint8 — hashing
    alpha: NDArrayF | None  # (H, W) float32 in [0, 1], or None
    mask: npt.NDArray[np.bool_] | None  # (H, W) bool when is_binary_mask, else None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def _to_float01(img: npt.NDArray[Any]) -> NDArrayF:
    """Normalize any supported dtype to float32 in [0, 1]."""
    if img.dtype == np.bool_:
        return img.astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img.astype(np.float32), 0.0, 1.0)
    if np.issubdtype(img.dtype, np.integer):
        # Unusual integer dtype: fall back to its positive range.
        info = np.iinfo(img.dtype)
        denom = float(max(info.max, 1))
        scaled = np.clip(img.astype(np.float32) / denom, 0.0, 1.0)
        return np.asarray(scaled, dtype=np.float32)
    raise TypeError(f"unsupported image dtype for canonicalization: {img.dtype}")


def _detect_binary_single_channel(single: npt.NDArray[Any]) -> bool:
    """True iff a single-channel image takes at most two values, one of them zero/min."""
    uniq = np.unique(single)
    if uniq.size == 0 or uniq.size > 2:
        return False
    if uniq.size == 1:
        return False  # constant image is not a "mask" in the useful sense
    if single.dtype == np.bool_:
        return True
    lo = float(uniq.min())
    hi = float(uniq.max())
    if lo != 0.0:
        return False
    if single.dtype == np.uint8:
        return hi == 255.0
    if np.issubdtype(single.dtype, np.floating):
        return hi == 1.0
    return False


def canonicalize(img: npt.NDArray[Any]) -> Canonical:
    """Apply the canonicalization contract to a single skill output."""
    if img.ndim == 2:
        h, w = img.shape
        n_ch = 1
        single = img
        alpha_raw = None
    elif img.ndim == 3:
        h, w, n_ch = img.shape
        if n_ch == 1:
            single = img[:, :, 0]
            alpha_raw = None
        elif n_ch == 3:
            single = None
            alpha_raw = None
        elif n_ch == 4:
            single = None
            alpha_raw = img[:, :, 3]
        else:
            raise ValueError(f"unsupported channel count: {n_ch}")
    else:
        raise ValueError(f"expected 2D or 3D image, got ndim={img.ndim}")

    is_mask = single is not None and _detect_binary_single_channel(single)

    if single is not None:
        f = _to_float01(single)  # (H, W)
        rgb = np.repeat(f[:, :, None], 3, axis=2)
        alpha = None
        mask = (f > 0.5).astype(np.bool_) if is_mask else None
    elif alpha_raw is not None:  # RGBA
        rgba = _to_float01(img)  # (H, W, 4)
        alpha = rgba[:, :, 3].astype(np.float32)
        fg = rgba[:, :, :3]
        rgb = (fg * alpha[:, :, None] + MIDGRAY * (1.0 - alpha[:, :, None])).astype(np.float32)
        mask = None
    else:  # RGB
        rgb = _to_float01(img).astype(np.float32)
        alpha = None
        mask = None

    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    rgb_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)

    return Canonical(
        version=CANON_VERSION,
        height=int(h),
        width=int(w),
        n_input_channels=int(n_ch),
        is_binary_mask=bool(is_mask),
        rgb=rgb,
        rgb_u8=rgb_u8,
        alpha=alpha,
        mask=mask,
    )


def content_hash(c: Canonical) -> str:
    """SHA-256 of the canonical uint8 content (+ shape, mask flag, alpha, version).

    EXACT duplicates collide here; different shapes never do.
    """
    h = hashlib.sha256()
    h.update(c.version.encode("ascii"))
    h.update(np.asarray([c.height, c.width, int(c.is_binary_mask)], dtype=np.int64).tobytes())
    h.update(c.rgb_u8.tobytes())
    if c.alpha is not None:
        alpha_u8 = np.clip(np.rint(c.alpha * 255.0), 0, 255).astype(np.uint8)
        h.update(b"A")
        h.update(alpha_u8.tobytes())
    return h.hexdigest()


def same_shape(a: Canonical, b: Canonical) -> bool:
    return a.shape == b.shape


def max_abs_pixel_diff(a: Canonical, b: Canonical) -> float:
    """Worst-case L∞ over the RGB view, in [0, 1]; ``inf`` on shape mismatch (shape gate)."""
    if not same_shape(a, b):
        return float("inf")
    return float(np.max(np.abs(a.rgb - b.rgb)))


def alpha_linf(a: Canonical, b: Canonical) -> float | None:
    """Worst-case L∞ over alpha channels, or None if neither output carries alpha."""
    if a.alpha is None and b.alpha is None:
        return None
    if not same_shape(a, b):
        return float("inf")
    aa = a.alpha if a.alpha is not None else np.ones(a.shape, dtype=np.float32)
    bb = b.alpha if b.alpha is not None else np.ones(b.shape, dtype=np.float32)
    return float(np.max(np.abs(aa - bb)))


def mask_iou(a: Canonical, b: Canonical) -> float:
    """Intersection-over-union of two binary masks; ``nan`` if either is not a mask."""
    if a.mask is None or b.mask is None or not same_shape(a, b):
        return float("nan")
    inter = np.logical_and(a.mask, b.mask).sum(dtype=np.int64)
    union = np.logical_or(a.mask, b.mask).sum(dtype=np.int64)
    if union == 0:
        return 1.0  # both empty → identical
    return float(inter) / float(union)

"""Shared low-level helpers for skill implementations.

Convention: arrays are treated as **RGB** (PIL load order). All ops are deterministic given
their inputs. Geometric ops preserve channel count; colour ops coerce to 3-channel RGB.
"""

from __future__ import annotations

import cv2
import numpy as np

from viscurate.skills.model import Image

# BT.601 / BT.709 luma weights (CLAUDE.md names grayscale_bt601 vs _bt709 as a SEMANTIC pair).
BT601 = (0.299, 0.587, 0.114)
BT709 = (0.2126, 0.7152, 0.0722)


def to_u8(a: Image) -> Image:
    """Clip to [0, 255] and cast to uint8."""
    return np.clip(a, 0, 255).astype(np.uint8)


def as_rgb(img: Image) -> Image:
    """Coerce any supported input to contiguous HxWx3 uint8 RGB (drops alpha)."""
    a = img if img.dtype == np.uint8 else to_u8(img)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if a.ndim == 3:
        ch = a.shape[2]
        if ch == 1:
            return cv2.cvtColor(np.ascontiguousarray(a[:, :, 0]), cv2.COLOR_GRAY2RGB)
        if ch == 3:
            return np.ascontiguousarray(a)
        if ch == 4:
            return np.ascontiguousarray(a[:, :, :3])
    raise ValueError(f"unsupported image shape for as_rgb: {img.shape}")


def to_gray(img: Image, coeffs: tuple[float, float, float] = BT601) -> Image:
    """Weighted-luma grayscale, returning a single-channel HxW uint8 image."""
    rgb = as_rgb(img).astype(np.float32)
    g = rgb[:, :, 0] * coeffs[0] + rgb[:, :, 1] * coeffs[1] + rgb[:, :, 2] * coeffs[2]
    return to_u8(g)


def odd_ge1(k: int) -> int:
    """Nearest odd integer >= 1 (kernel sizes must be odd for several cv2 ops)."""
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def to_binary(img: Image, thresh: int = 127) -> Image:
    """Single-channel {0,255} mask from the luma channel (input for shape/region ops)."""
    g = to_gray(img)
    return ((g > thresh).astype(np.uint8)) * 255

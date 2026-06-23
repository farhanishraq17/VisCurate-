"""Deterministic, license-free synthetic probe generators (CLAUDE.md §2.1).

Every generator is a pure function of a :class:`numpy.random.Generator` (seeded from the root
via :class:`viscurate.rng.SeedManager`), so the battery is byte-reproducible. These cover all
diversity axes except true natural-photo statistics (which come from the COCO loader): the
domains (gradient/texture/shape/document/colorchart/noise/degenerate), the channel formats
(RGB/RGBA/grayscale/16-bit/palette), low- vs high-frequency signal, and the degenerate cases.

Probes are stored as raw arrays (`.npy`), so dtype and channel count survive exactly — a
16-bit or RGBA probe reaches the skills unchanged. ``palette`` probes are index arrays with a
small number of distinct values (the palette LUT itself is not modelled in v1).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from viscurate.probes.manifest import ChannelFormat, Domain
from viscurate.rng import SeedManager
from viscurate.skills.model import Image as Array

__all__ = ["GeneratedProbe", "generate_synthetic_probes"]


@dataclass(frozen=True)
class GeneratedProbe:
    """A synthetic probe before it is hashed/written and turned into a manifest entry."""

    base_id: str
    array: Array
    domain: Domain
    channel_format: ChannelFormat
    notes: str = ""


# --- low-level builders ----------------------------------------------------------------


def _linear_gradient(g: np.random.Generator, h: int, w: int) -> Array:
    c0 = g.integers(0, 256, 3)
    c1 = g.integers(0, 256, 3)
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    row = (c0[None, None, :] * (1 - t) + c1[None, None, :] * t).astype(np.uint8)
    return np.repeat(row, h, axis=0)


def _radial_gradient(g: np.random.Generator, h: int, w: int) -> Array:
    c0 = g.integers(0, 256, 3).astype(np.float32)
    c1 = g.integers(0, 256, 3).astype(np.float32)
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    r = np.clip(r, 0, 1).astype(np.float32)[:, :, None]
    return (c0[None, None, :] * (1 - r) + c1[None, None, :] * r).astype(np.uint8)


def _angular_gradient(g: np.random.Generator, h: int, w: int) -> Array:
    yy, xx = np.ogrid[:h, :w]
    ang = (np.arctan2(yy - h / 2, xx - w / 2) + np.pi) / (2 * np.pi)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:, :, 0] = (ang * 179).astype(np.uint8)
    hsv[:, :, 1] = 200
    hsv[:, :, 2] = 255
    import cv2

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _value_noise(g: np.random.Generator, h: int, w: int) -> Array:
    import cv2

    s = int(g.integers(4, 16))
    small = g.integers(0, 256, (max(1, h // s), max(1, w // s), 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _white_noise(g: np.random.Generator, h: int, w: int) -> Array:
    return g.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _checkerboard(g: np.random.Generator, h: int, w: int) -> Array:
    n = int(g.integers(4, 16))
    ys = (np.arange(h) // max(1, h // n))[:, None]
    xs = (np.arange(w) // max(1, w // n))[None, :]
    board = (((ys + xs) % 2) * 255).astype(np.uint8)
    return np.repeat(board[:, :, None], 3, axis=2)


def _stripes(g: np.random.Generator, h: int, w: int) -> Array:
    freq = int(g.integers(4, 24))
    phase = float(g.random())
    x = np.linspace(0, freq * np.pi, w, dtype=np.float32) + phase
    line = ((np.sin(x) > 0) * 255).astype(np.uint8)
    img = np.tile(line, (h, 1))
    return np.repeat(img[:, :, None], 3, axis=2)


def _gabor(g: np.random.Generator, h: int, w: int) -> Array:
    import cv2

    k = max(h, w) | 1
    theta = float(g.random()) * np.pi
    kern = cv2.getGaborKernel((k, k), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kern = (kern - kern.min()) / (np.ptp(kern) + 1e-9)
    crop = (kern[:h, :w] * 255).astype(np.uint8)
    return np.repeat(crop[:, :, None], 3, axis=2)


def _shapes(g: np.random.Generator, h: int, w: int) -> Array:
    import cv2

    img = np.full((h, w, 3), int(g.integers(0, 256)), np.uint8)
    for _ in range(int(g.integers(3, 9))):
        color = tuple(int(c) for c in g.integers(0, 256, 3))
        kind = int(g.integers(0, 3))
        if kind == 0:
            c = (int(g.integers(0, w)), int(g.integers(0, h)))
            cv2.circle(img, c, int(g.integers(3, max(4, min(h, w) // 3))), color, -1)
        elif kind == 1:
            p0 = (int(g.integers(0, w)), int(g.integers(0, h)))
            p1 = (int(g.integers(0, w)), int(g.integers(0, h)))
            cv2.rectangle(img, p0, p1, color, -1)
        else:
            p0 = (int(g.integers(0, w)), int(g.integers(0, h)))
            p1 = (int(g.integers(0, w)), int(g.integers(0, h)))
            cv2.line(img, p0, p1, color, int(g.integers(1, 5)))
    return img


_WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "probe",
    "image",
    "skill",
    "curate",
    "verify",
    "output",
    "ground",
    "truth",
]


def _document(g: np.random.Generator, h: int, w: int) -> Array:
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 6
    while y < h - 12:
        n = int(g.integers(3, 9))
        text = " ".join(_WORDS[int(g.integers(0, len(_WORDS)))] for _ in range(n))
        draw.text((6, y), text, fill=(0, 0, 0), font=font)
        if g.random() < 0.2:
            draw.rectangle([6, y - 2, w - 6, y + 12], outline=(0, 0, 0))
        y += int(g.integers(12, 22))
    return np.asarray(img, dtype=np.uint8)


def _color_chart(g: np.random.Generator, h: int, w: int) -> Array:
    cols, rows = 6, 4
    img = np.zeros((h, w, 3), np.uint8)
    for r in range(rows):
        for c in range(cols):
            color = g.integers(0, 256, 3)
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            img[y0:y1, x0:x1] = color
    return img


# --- format converters -----------------------------------------------------------------


def _to_rgba(rgb: Array, g: np.random.Generator) -> Array:
    h, w = rgb.shape[:2]
    alpha = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, axis=0)
    return np.dstack([rgb, alpha])


def _to_gray(rgb: Array) -> Array:
    import cv2

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _to_gray16(rgb: Array) -> Array:
    gray = _to_gray(rgb).astype(np.uint16)
    return (gray * 257).astype(np.uint16)  # stretch 0..255 -> 0..65535


def _to_palette(rgb: Array, levels: int = 8) -> Array:
    gray = _to_gray(rgb).astype(np.float32)
    return (np.round(gray / 255.0 * (levels - 1))).astype(np.uint8)  # index array


# --- degenerate cases ------------------------------------------------------------------


def _degenerate() -> Iterator[GeneratedProbe]:
    yield GeneratedProbe(
        "deg_black", np.zeros((64, 64, 3), np.uint8), "degenerate", "rgb", "all_black"
    )
    yield GeneratedProbe(
        "deg_white", np.full((64, 64, 3), 255, np.uint8), "degenerate", "rgb", "all_white"
    )
    yield GeneratedProbe(
        "deg_single", np.full((64, 64, 3), 73, np.uint8), "degenerate", "rgb", "single_color"
    )
    yield GeneratedProbe(
        "deg_1x1", np.full((1, 1, 3), 128, np.uint8), "degenerate", "rgb", "one_by_one"
    )
    yield GeneratedProbe(
        "deg_thin",
        np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (1, 1, 3)),
        "degenerate",
        "rgb",
        "thin_1xN",
    )
    yield GeneratedProbe(
        "deg_tall",
        np.tile(np.arange(64, dtype=np.uint8)[:, None, None], (1, 1, 3)),
        "degenerate",
        "rgb",
        "thin_Nx1",
    )
    big = np.zeros((1024, 1024, 3), np.uint8)
    big[:, :, 0] = np.linspace(0, 255, 1024, dtype=np.uint8)[None, :]
    yield GeneratedProbe("deg_highres", big, "degenerate", "rgb", "high_res_1024")


# --- the generator ---------------------------------------------------------------------

_Gen = Callable[[np.random.Generator, int, int], Array]
_RGB_GENERATORS: dict[Domain, tuple[_Gen, ...]] = {
    "gradient": (_linear_gradient, _radial_gradient, _angular_gradient),
    "texture": (_value_noise, _checkerboard, _stripes, _gabor),
    "shape": (_shapes,),
    "document": (_document,),
    "colorchart": (_color_chart,),
    "noise": (_white_noise,),
}


def generate_synthetic_probes(
    sm: SeedManager, counts: dict[str, int], *, size: int = 128
) -> list[GeneratedProbe]:
    """Generate the synthetic battery: ``counts`` per domain, plus format variants + degenerates.

    A fraction of each domain's probes are emitted in non-RGB formats (RGBA / grayscale /
    16-bit / palette) so every required channel format is exercised.
    """
    out: list[GeneratedProbe] = []
    for domain, gens in _RGB_GENERATORS.items():
        n = counts.get(domain, 0)
        for i in range(n):
            g = sm.generator("synthetic", domain, i)
            fn = gens[i % len(gens)]
            rgb = fn(g, size, size)
            fmt: ChannelFormat = "rgb"
            arr: Array = rgb
            notes = fn.__name__.lstrip("_")
            # Spread channel formats deterministically across the domain's probes.
            sel = i % 5
            if sel == 1:
                arr, fmt = _to_rgba(rgb, g), "rgba"
            elif sel == 2:
                arr, fmt = _to_gray(rgb), "gray"
            elif sel == 3:
                arr, fmt = _to_gray16(rgb), "gray16"
            elif sel == 4:
                arr, fmt = _to_palette(rgb), "palette"
            out.append(
                GeneratedProbe(
                    f"syn_{domain}_{i:03d}", np.ascontiguousarray(arr), domain, fmt, notes
                )
            )
    out.extend(_degenerate())
    return out

"""Geometric / canvas skills (CLAUDE.md families 1–25).

These preserve channel count (so alpha/grayscale survive) and are deterministic. The set
deliberately includes the planted relations the later phases lean on:

* ``rotate_90 / 180 / 270 ⊑ rotate_canvas_degrees`` and ``rotate_45 ⊑ rotate_canvas_degrees``
  (subsumption, the last with interpolation),
* ``crop_center_percentage ⊑ crop_bounding_box`` (subsumption),
* ``rotate_90`` vs ``transpose`` and ``resize_*`` nearest/bilinear/bicubic and
  ``pad_reflect`` vs ``pad_replicate`` (engineered hard negatives / DISTINCT),
* ``random_crop`` is a seeded-stochastic determinism probe.
"""

from __future__ import annotations

import cv2
import numpy as np

from viscurate.skills.library._build import make_skill, param
from viscurate.skills.model import Image, Params, Skill


def flip_horizontal(image: Image, params: Params, seed: int) -> Image:
    return np.ascontiguousarray(image[:, ::-1])


def flip_vertical(image: Image, params: Params, seed: int) -> Image:
    return np.ascontiguousarray(image[::-1, :])


def rotate_90(image: Image, params: Params, seed: int) -> Image:
    return np.ascontiguousarray(np.rot90(image, 1))  # counter-clockwise


def rotate_180(image: Image, params: Params, seed: int) -> Image:
    return np.ascontiguousarray(np.rot90(image, 2))


def rotate_270(image: Image, params: Params, seed: int) -> Image:
    return np.ascontiguousarray(np.rot90(image, 3))


def transpose(image: Image, params: Params, seed: int) -> Image:
    axes = (1, 0) if image.ndim == 2 else (1, 0, 2)
    return np.ascontiguousarray(np.transpose(image, axes))


def rotate_canvas_degrees(image: Image, params: Params, seed: int) -> Image:
    """Rotate CCW by `degrees`, expanding the canvas to fit. Exact for multiples of 90."""
    deg = float(params["degrees"])
    if deg % 90 == 0:
        return np.ascontiguousarray(np.rot90(image, int(deg // 90) % 4))
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += nw / 2.0 - cx
    matrix[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(
        image, matrix, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


def rotate_45(image: Image, params: Params, seed: int) -> Image:
    """Fixed 45° specialization of rotate_canvas_degrees (a subsumption host)."""
    return rotate_canvas_degrees(image, {"degrees": 45.0}, seed)


def translate(image: Image, params: Params, seed: int) -> Image:
    h, w = image.shape[:2]
    matrix = np.array(
        [[1.0, 0.0, float(params["dx_frac"]) * w], [0.0, 1.0, float(params["dy_frac"]) * h]],
        dtype=np.float32,
    )
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT)


def shear_horizontal(image: Image, params: Params, seed: int) -> Image:
    h, w = image.shape[:2]
    s = float(params["shear"])
    matrix = np.array([[1.0, s, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    nw = int(w + abs(s) * h)
    return cv2.warpAffine(image, matrix, (max(1, nw), h), borderMode=cv2.BORDER_CONSTANT)


def shear_vertical(image: Image, params: Params, seed: int) -> Image:
    h, w = image.shape[:2]
    s = float(params["shear"])
    matrix = np.array([[1.0, 0.0, 0.0], [s, 1.0, 0.0]], dtype=np.float32)
    nh = int(h + abs(s) * w)
    return cv2.warpAffine(image, matrix, (w, max(1, nh)), borderMode=cv2.BORDER_CONSTANT)


def scale_xy(image: Image, params: Params, seed: int) -> Image:
    """Anisotropic rescale (independent x/y factors)."""
    h, w = image.shape[:2]
    nw = max(1, int(round(w * float(params["sx"]))))
    nh = max(1, int(round(h * float(params["sy"]))))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)


def crop_center_percentage(image: Image, params: Params, seed: int) -> Image:
    pct = float(params["percent"]) / 100.0
    h, w = image.shape[:2]
    ch, cw = max(1, int(round(h * pct))), max(1, int(round(w * pct)))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return np.ascontiguousarray(image[y0 : y0 + ch, x0 : x0 + cw])


def crop_bounding_box(image: Image, params: Params, seed: int) -> Image:
    """Crop a fractional box (left, top, width, height ∈ [0,1]); clamped to the image."""
    h, w = image.shape[:2]
    left = float(params["left"])
    top = float(params["top"])
    width = float(params["width"])
    height = float(params["height"])
    cw = int(round(width * w))
    ch = int(round(height * h))
    centered_x = abs(left - (1.0 - width) / 2.0) <= 1.0e-9
    centered_y = abs(top - (1.0 - height) / 2.0) <= 1.0e-9
    x0 = (w - cw) // 2 if centered_x else int(round(left * w))
    y0 = (h - ch) // 2 if centered_y else int(round(top * h))
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    cw = max(1, min(cw, w - x0))
    ch = max(1, min(ch, h - y0))
    return np.ascontiguousarray(image[y0 : y0 + ch, x0 : x0 + cw])


def random_crop(image: Image, params: Params, seed: int) -> Image:
    """Seeded-stochastic: a fixed-size crop at a seed-determined location."""
    h, w = image.shape[:2]
    frac = float(params["size_frac"])
    ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
    rng = np.random.default_rng(seed)
    y0 = int(rng.integers(0, max(1, h - ch + 1)))
    x0 = int(rng.integers(0, max(1, w - cw + 1)))
    return np.ascontiguousarray(image[y0 : y0 + ch, x0 : x0 + cw])


def zoom_in_center(image: Image, params: Params, seed: int) -> Image:
    """Crop the central 1/factor region then resize back to the original shape."""
    factor = float(params["factor"])
    h, w = image.shape[:2]
    ch, cw = max(1, int(h / factor)), max(1, int(w / factor))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = image[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _resize(image: Image, scale: float, interp: int) -> Image:
    h, w = image.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(image, (nw, nh), interpolation=interp)


def resize_nearest(image: Image, params: Params, seed: int) -> Image:
    return _resize(image, float(params["scale"]), cv2.INTER_NEAREST)


def resize_bilinear(image: Image, params: Params, seed: int) -> Image:
    return _resize(image, float(params["scale"]), cv2.INTER_LINEAR)


def resize_bicubic(image: Image, params: Params, seed: int) -> Image:
    return _resize(image, float(params["scale"]), cv2.INTER_CUBIC)


def resize_fixed(image: Image, params: Params, seed: int) -> Image:
    """Resize to an absolute target size (width, height)."""
    return cv2.resize(
        image, (int(params["width"]), int(params["height"])), interpolation=cv2.INTER_LINEAR
    )


def _pad(image: Image, border: int, mode: int, value: int = 0) -> Image:
    return cv2.copyMakeBorder(image, border, border, border, border, mode, value=value)


def pad_reflect(image: Image, params: Params, seed: int) -> Image:
    return _pad(image, int(params["border"]), cv2.BORDER_REFLECT)


def pad_replicate(image: Image, params: Params, seed: int) -> Image:
    return _pad(image, int(params["border"]), cv2.BORDER_REPLICATE)


def pad_constant(image: Image, params: Params, seed: int) -> Image:
    return _pad(image, int(params["border"]), cv2.BORDER_CONSTANT, int(params["value"]))


def pad_to_square(image: Image, params: Params, seed: int) -> Image:
    """Pad the shorter side with a constant so the canvas becomes square."""
    h, w = image.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    bottom, right = side - h - top, side - w - left
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=int(params["value"])
    )


def tile_2x2(image: Image, params: Params, seed: int) -> Image:
    """Replicate the image in a 2×2 grid."""
    row = np.concatenate([image, image], axis=1)
    return np.ascontiguousarray(np.concatenate([row, row], axis=0))


def build() -> list[Skill]:
    return [
        make_skill(
            "flip_horizontal_v1",
            "Flip horizontal",
            "Mirror the image left-to-right.",
            flip_horizontal,
            "geometric",
        ),
        make_skill(
            "flip_vertical_v1",
            "Flip vertical",
            "Mirror the image top-to-bottom.",
            flip_vertical,
            "geometric",
        ),
        make_skill(
            "rotate_90_v1",
            "Rotate 90°",
            "Rotate 90 degrees counter-clockwise.",
            rotate_90,
            "geometric",
        ),
        make_skill("rotate_180_v1", "Rotate 180°", "Rotate 180 degrees.", rotate_180, "geometric"),
        make_skill(
            "rotate_270_v1",
            "Rotate 270°",
            "Rotate 270 degrees counter-clockwise.",
            rotate_270,
            "geometric",
        ),
        make_skill(
            "transpose_v1",
            "Transpose",
            "Swap the row and column axes (matrix transpose).",
            transpose,
            "geometric",
        ),
        make_skill(
            "rotate_canvas_degrees_v1",
            "Rotate (arbitrary)",
            "Rotate counter-clockwise by an arbitrary angle, expanding the canvas.",
            rotate_canvas_degrees,
            "geometric",
            (param("degrees", "float", 0.0, minimum=-360.0, maximum=360.0),),
        ),
        make_skill(
            "rotate_45_v1",
            "Rotate 45°",
            "Rotate 45 degrees counter-clockwise (fixed-angle specialization).",
            rotate_45,
            "geometric",
        ),
        make_skill(
            "translate_v1",
            "Translate",
            "Shift the image by a fraction of its size.",
            translate,
            "geometric",
            (
                param("dx_frac", "float", 0.0, minimum=-1.0, maximum=1.0),
                param("dy_frac", "float", 0.0, minimum=-1.0, maximum=1.0),
            ),
        ),
        make_skill(
            "shear_horizontal_v1",
            "Shear horizontal",
            "Apply a horizontal shear.",
            shear_horizontal,
            "geometric",
            (param("shear", "float", 0.2, minimum=-2.0, maximum=2.0),),
        ),
        make_skill(
            "shear_vertical_v1",
            "Shear vertical",
            "Apply a vertical shear.",
            shear_vertical,
            "geometric",
            (param("shear", "float", 0.2, minimum=-2.0, maximum=2.0),),
        ),
        make_skill(
            "scale_xy_v1",
            "Scale (anisotropic)",
            "Rescale x and y by independent factors.",
            scale_xy,
            "geometric",
            (
                param("sx", "float", 1.0, minimum=0.05, maximum=8.0),
                param("sy", "float", 1.0, minimum=0.05, maximum=8.0),
            ),
        ),
        make_skill(
            "crop_center_percentage_v1",
            "Crop center %",
            "Keep a centered region covering `percent`% of each dimension.",
            crop_center_percentage,
            "geometric",
            (param("percent", "float", 50.0, minimum=1.0, maximum=100.0),),
        ),
        make_skill(
            "crop_bounding_box_v1",
            "Crop bounding box",
            "Crop a fractional bounding box (left, top, width, height).",
            crop_bounding_box,
            "geometric",
            (
                param("left", "float", 0.25, minimum=0.0, maximum=1.0),
                param("top", "float", 0.25, minimum=0.0, maximum=1.0),
                param("width", "float", 0.5, minimum=0.0, maximum=1.0),
                param("height", "float", 0.5, minimum=0.0, maximum=1.0),
            ),
        ),
        make_skill(
            "random_crop_v1",
            "Random crop",
            "Crop a fixed-size region at a seed-determined location.",
            random_crop,
            "geometric",
            (param("size_frac", "float", 0.5, minimum=0.05, maximum=1.0),),
            seeded_stochastic=True,
        ),
        make_skill(
            "zoom_in_center_v1",
            "Zoom (center)",
            "Crop the central region and resize back to the original shape.",
            zoom_in_center,
            "geometric",
            (param("factor", "float", 2.0, minimum=1.0, maximum=8.0),),
        ),
        make_skill(
            "resize_nearest_v1",
            "Resize (nearest)",
            "Rescale using nearest-neighbour interpolation.",
            resize_nearest,
            "geometric",
            (param("scale", "float", 0.5, minimum=0.05, maximum=8.0),),
        ),
        make_skill(
            "resize_bilinear_v1",
            "Resize (bilinear)",
            "Rescale using bilinear interpolation.",
            resize_bilinear,
            "geometric",
            (param("scale", "float", 0.5, minimum=0.05, maximum=8.0),),
        ),
        make_skill(
            "resize_bicubic_v1",
            "Resize (bicubic)",
            "Rescale using bicubic interpolation.",
            resize_bicubic,
            "geometric",
            (param("scale", "float", 0.5, minimum=0.05, maximum=8.0),),
        ),
        make_skill(
            "resize_fixed_v1",
            "Resize (fixed)",
            "Resize to an absolute width and height.",
            resize_fixed,
            "geometric",
            (
                param("width", "int", 128, minimum=1, maximum=4096),
                param("height", "int", 128, minimum=1, maximum=4096),
            ),
        ),
        make_skill(
            "pad_reflect_v1",
            "Pad (reflect)",
            "Add a reflected border of `border` pixels.",
            pad_reflect,
            "geometric",
            (param("border", "int", 8, minimum=0, maximum=256),),
        ),
        make_skill(
            "pad_replicate_v1",
            "Pad (replicate)",
            "Add a replicated (edge-clamped) border.",
            pad_replicate,
            "geometric",
            (param("border", "int", 8, minimum=0, maximum=256),),
        ),
        make_skill(
            "pad_constant_v1",
            "Pad (constant)",
            "Add a constant-valued border.",
            pad_constant,
            "geometric",
            (
                param("border", "int", 8, minimum=0, maximum=256),
                param("value", "int", 0, minimum=0, maximum=255),
            ),
        ),
        make_skill(
            "pad_to_square_v1",
            "Pad to square",
            "Pad the shorter side so the canvas becomes square.",
            pad_to_square,
            "geometric",
            (param("value", "int", 0, minimum=0, maximum=255),),
        ),
        make_skill(
            "tile_2x2_v1", "Tile 2×2", "Replicate the image in a 2×2 grid.", tile_2x2, "geometric"
        ),
    ]

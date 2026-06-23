"""Colour / exposure / contrast skills (CLAUDE.md families 26–50).

Planted relations exercised later: ``grayscale_bt601`` vs ``grayscale_bt709`` and
``equalize_histogram_global`` vs ``equalize_clahe`` and ``gamma_correct`` vs
``exposure_stops`` (SEMANTIC pairs); ``linear_contrast_stretch ⊑
contrast_stretching_percentile`` and ``linear_contrast_stretch ⊑ levels_adjust``
(subsumption); ``posterize`` vs ``quantize_uniform`` (near-duplicate hard cases).
"""

from __future__ import annotations

import cv2
import numpy as np

from viscurate.skills.library._build import make_skill, param
from viscurate.skills.library._ops import BT601, BT709, as_rgb, to_gray, to_u8
from viscurate.skills.model import Image, Params, Skill


def grayscale_bt601(image: Image, params: Params, seed: int) -> Image:
    return to_gray(image, BT601)


def grayscale_bt709(image: Image, params: Params, seed: int) -> Image:
    return to_gray(image, BT709)


def invert(image: Image, params: Params, seed: int) -> Image:
    return to_u8(255 - as_rgb(image).astype(np.int16))


def brightness_add(image: Image, params: Params, seed: int) -> Image:
    return to_u8(as_rgb(image).astype(np.float32) + float(params["delta"]))


def brightness_multiply(image: Image, params: Params, seed: int) -> Image:
    return to_u8(as_rgb(image).astype(np.float32) * float(params["gain"]))


def exposure_stops(image: Image, params: Params, seed: int) -> Image:
    """Multiply linear intensity by 2**stops (a photographic-stops view of exposure)."""
    return to_u8(as_rgb(image).astype(np.float32) * (2.0 ** float(params["stops"])))


def contrast_scale(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).astype(np.float32)
    return to_u8((rgb - 127.5) * float(params["factor"]) + 127.5)


def gamma_correct(image: Image, params: Params, seed: int) -> Image:
    gamma = float(params["gamma"])
    lut = to_u8(((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255.0)
    return cv2.LUT(as_rgb(image), lut)


def saturation_gain(image: Image, params: Params, seed: int) -> Image:
    hsv = cv2.cvtColor(as_rgb(image), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(params["factor"]), 0, 255)
    return cv2.cvtColor(to_u8(hsv), cv2.COLOR_HSV2RGB)


def desaturate_partial(image: Image, params: Params, seed: int) -> Image:
    """Blend toward grayscale by `amount` (0 = unchanged, 1 = fully gray)."""
    rgb = as_rgb(image).astype(np.float32)
    gray = to_gray(image).astype(np.float32)[:, :, None]
    a = float(params["amount"])
    return to_u8(rgb * (1.0 - a) + gray * a)


def hue_shift(image: Image, params: Params, seed: int) -> Image:
    hsv = cv2.cvtColor(as_rgb(image), cv2.COLOR_RGB2HSV)
    shift = int(round(float(params["degrees"]) / 2.0)) % 180  # cv2 hue is 0..179
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 180
    return cv2.cvtColor(to_u8(hsv), cv2.COLOR_HSV2RGB)


def sepia_tone(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).astype(np.float32)
    matrix = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        dtype=np.float32,
    )
    return to_u8(rgb @ matrix.T)


def color_temperature_shift(image: Image, params: Params, seed: int) -> Image:
    """Warm (+) / cool (-): push red up and blue down."""
    rgb = as_rgb(image).astype(np.float32)
    amt = float(params["amount"])
    rgb[:, :, 0] += amt
    rgb[:, :, 2] -= amt
    return to_u8(rgb)


def color_balance_rgb(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).astype(np.float32)
    rgb[:, :, 0] *= float(params["r_gain"])
    rgb[:, :, 1] *= float(params["g_gain"])
    rgb[:, :, 2] *= float(params["b_gain"])
    return to_u8(rgb)


def channel_isolate(image: Image, params: Params, seed: int) -> Image:
    """Keep one channel, zero the others."""
    rgb = as_rgb(image)
    idx = {"red": 0, "green": 1, "blue": 2}[str(params["channel"])]
    out = np.zeros_like(rgb)
    out[:, :, idx] = rgb[:, :, idx]
    return out


def channel_swap_rgb_bgr(image: Image, params: Params, seed: int) -> Image:
    return np.ascontiguousarray(as_rgb(image)[:, :, ::-1])


def threshold_binary(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image)
    return ((gray > int(params["thresh"])).astype(np.uint8)) * 255  # HxW binary mask


def solarize(image: Image, params: Params, seed: int) -> Image:
    """Invert pixels at or above a threshold (the Sabattier effect)."""
    rgb = as_rgb(image)
    t = int(params["thresh"])
    return np.where(rgb >= t, 255 - rgb, rgb).astype(np.uint8)


def posterize(image: Image, params: Params, seed: int) -> Image:
    bits = int(params["bits"])
    shift = 8 - bits
    rgb = as_rgb(image)
    return np.left_shift(np.right_shift(rgb, shift), shift)


def quantize_uniform(image: Image, params: Params, seed: int) -> Image:
    """Quantize each channel to `levels` evenly-spaced values."""
    n = max(2, int(params["levels"]))
    rgb = as_rgb(image).astype(np.float32)
    return to_u8(np.round(rgb / 255.0 * (n - 1)) / (n - 1) * 255.0)


def linear_contrast_stretch(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).astype(np.float32)
    lo, hi = float(rgb.min()), float(rgb.max())
    if hi <= lo:
        return to_u8(rgb)
    return to_u8((rgb - lo) * 255.0 / (hi - lo))


def contrast_stretching_percentile(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).astype(np.float32)
    lo = float(np.percentile(rgb, float(params["low_pct"])))
    hi = float(np.percentile(rgb, float(params["high_pct"])))
    if hi <= lo:
        return to_u8(rgb)
    return to_u8(np.clip((rgb - lo) * 255.0 / (hi - lo), 0, 255))


def levels_adjust(image: Image, params: Params, seed: int) -> Image:
    """Map [black, white] to [0, 255] (a generalization of linear stretch)."""
    rgb = as_rgb(image).astype(np.float32)
    b, w = float(params["black"]), float(params["white"])
    if w <= b:
        return to_u8(rgb)
    return to_u8(np.clip((rgb - b) * 255.0 / (w - b), 0, 255))


def equalize_histogram_global(image: Image, params: Params, seed: int) -> Image:
    ycc = cv2.cvtColor(as_rgb(image), cv2.COLOR_RGB2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def equalize_clahe(image: Image, params: Params, seed: int) -> Image:
    ycc = cv2.cvtColor(as_rgb(image), cv2.COLOR_RGB2YCrCb)
    grid = max(1, int(params["grid"]))
    clahe = cv2.createCLAHE(clipLimit=float(params["clip"]), tileGridSize=(grid, grid))
    ycc[:, :, 0] = clahe.apply(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def build() -> list[Skill]:
    return [
        make_skill(
            "grayscale_bt601_v1",
            "Grayscale (BT.601)",
            "Luma grayscale using BT.601 weights.",
            grayscale_bt601,
            "color",
        ),
        make_skill(
            "grayscale_bt709_v1",
            "Grayscale (BT.709)",
            "Luma grayscale using BT.709 weights.",
            grayscale_bt709,
            "color",
        ),
        make_skill("invert_v1", "Invert", "Photographic negative.", invert, "color"),
        make_skill(
            "brightness_add_v1",
            "Brightness (add)",
            "Add a constant offset to every channel.",
            brightness_add,
            "color",
            (param("delta", "float", 0.0, minimum=-255.0, maximum=255.0),),
        ),
        make_skill(
            "brightness_multiply_v1",
            "Brightness (multiply)",
            "Multiply every channel by a gain.",
            brightness_multiply,
            "color",
            (param("gain", "float", 1.0, minimum=0.0, maximum=8.0),),
        ),
        make_skill(
            "exposure_stops_v1",
            "Exposure (stops)",
            "Adjust exposure by photographic stops (×2**stops).",
            exposure_stops,
            "color",
            (param("stops", "float", 0.0, minimum=-4.0, maximum=4.0),),
        ),
        make_skill(
            "contrast_scale_v1",
            "Contrast",
            "Scale contrast about mid-gray.",
            contrast_scale,
            "color",
            (param("factor", "float", 1.0, minimum=0.0, maximum=4.0),),
        ),
        make_skill(
            "gamma_correct_v1",
            "Gamma",
            "Apply a gamma transfer curve.",
            gamma_correct,
            "color",
            (param("gamma", "float", 1.0, minimum=0.1, maximum=5.0),),
        ),
        make_skill(
            "saturation_gain_v1",
            "Saturation",
            "Multiply HSV saturation by a factor.",
            saturation_gain,
            "color",
            (param("factor", "float", 1.0, minimum=0.0, maximum=4.0),),
        ),
        make_skill(
            "desaturate_partial_v1",
            "Desaturate (partial)",
            "Blend toward grayscale by an amount.",
            desaturate_partial,
            "color",
            (param("amount", "float", 0.5, minimum=0.0, maximum=1.0),),
        ),
        make_skill(
            "hue_shift_v1",
            "Hue shift",
            "Rotate hue by a number of degrees.",
            hue_shift,
            "color",
            (param("degrees", "float", 0.0, minimum=-180.0, maximum=180.0),),
        ),
        make_skill("sepia_tone_v1", "Sepia", "Apply a sepia colour matrix.", sepia_tone, "color"),
        make_skill(
            "color_temperature_shift_v1",
            "Colour temperature",
            "Warm/cool the image (red up, blue down).",
            color_temperature_shift,
            "color",
            (param("amount", "float", 20.0, minimum=-128.0, maximum=128.0),),
        ),
        make_skill(
            "color_balance_rgb_v1",
            "Colour balance",
            "Per-channel gain (R, G, B).",
            color_balance_rgb,
            "color",
            (
                param("r_gain", "float", 1.0, minimum=0.0, maximum=4.0),
                param("g_gain", "float", 1.0, minimum=0.0, maximum=4.0),
                param("b_gain", "float", 1.0, minimum=0.0, maximum=4.0),
            ),
        ),
        make_skill(
            "channel_isolate_v1",
            "Channel isolate",
            "Keep one RGB channel, zero the others.",
            channel_isolate,
            "color",
            (param("channel", "enum", "red", choices=("red", "green", "blue")),),
        ),
        make_skill(
            "channel_swap_rgb_bgr_v1",
            "Channel swap",
            "Reverse channel order (RGB↔BGR).",
            channel_swap_rgb_bgr,
            "color",
        ),
        make_skill(
            "threshold_binary_v1",
            "Threshold",
            "Binarize the luma channel at a threshold.",
            threshold_binary,
            "color",
            (param("thresh", "int", 128, minimum=0, maximum=255),),
        ),
        make_skill(
            "solarize_v1",
            "Solarize",
            "Invert pixels at or above a threshold.",
            solarize,
            "color",
            (param("thresh", "int", 128, minimum=0, maximum=255),),
        ),
        make_skill(
            "posterize_v1",
            "Posterize",
            "Reduce to `bits` bits per channel.",
            posterize,
            "color",
            (param("bits", "int", 4, minimum=1, maximum=8),),
        ),
        make_skill(
            "quantize_uniform_v1",
            "Quantize (uniform)",
            "Quantize each channel to N evenly-spaced levels.",
            quantize_uniform,
            "color",
            (param("levels", "int", 8, minimum=2, maximum=64),),
        ),
        make_skill(
            "linear_contrast_stretch_v1",
            "Linear stretch",
            "Stretch the global min/max to the full range.",
            linear_contrast_stretch,
            "color",
        ),
        make_skill(
            "contrast_stretching_percentile_v1",
            "Percentile stretch",
            "Stretch between low/high percentiles, clipping outliers.",
            contrast_stretching_percentile,
            "color",
            (
                param("low_pct", "float", 2.0, minimum=0.0, maximum=49.0),
                param("high_pct", "float", 98.0, minimum=51.0, maximum=100.0),
            ),
        ),
        make_skill(
            "levels_adjust_v1",
            "Levels",
            "Map a [black, white] window to the full range.",
            levels_adjust,
            "color",
            (
                param("black", "float", 0.0, minimum=0.0, maximum=254.0),
                param("white", "float", 255.0, minimum=1.0, maximum=255.0),
            ),
        ),
        make_skill(
            "equalize_histogram_global_v1",
            "Equalize (global)",
            "Global histogram equalization of the luma channel.",
            equalize_histogram_global,
            "color",
        ),
        make_skill(
            "equalize_clahe_v1",
            "Equalize (CLAHE)",
            "Contrast-limited adaptive histogram equalization.",
            equalize_clahe,
            "color",
            (
                param("clip", "float", 2.0, minimum=0.5, maximum=40.0),
                param("grid", "int", 8, minimum=1, maximum=32),
            ),
        ),
    ]

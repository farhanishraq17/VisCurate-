"""Masks / reconstruction / synthesis skills (CLAUDE.md families 76–100).

This family hosts several of the trickier planted cases:

* ``inpaint_telea`` vs ``inpaint_ns`` — two algorithms for the same job (a SEMANTIC pair),
* ``threshold_otsu`` vs ``threshold_adaptive_mean`` — same intent, different method,
* ``palette_reduce_kmeans`` is **platform-sensitive** (``cv2.kmeans``; §1.4) and seeded,
* ``mask_to_rgba`` emits an **RGBA** output to exercise the alpha path of canonicalization,
* ``value_noise_synthesize`` is a **seeded-stochastic** synthesis probe.

All ops are deterministic given ``(image, params, seed)`` on a fixed platform.
"""

from __future__ import annotations

import cv2
import numpy as np

from viscurate.skills.library._build import make_skill, param
from viscurate.skills.library._ops import as_rgb, odd_ge1, to_binary, to_gray, to_u8
from viscurate.skills.model import Image, Params, Skill


def _highlight_mask(rgb: Image, pct: float = 97.0) -> Image:
    gray = to_gray(rgb)
    thr = float(np.percentile(gray, pct))
    return ((gray >= thr).astype(np.uint8)) * 255


def _inpaintable(rgb: Image) -> bool:
    # cv2 inpainting reads a neighbourhood band; on degenerate thin/tiny images it accesses
    # ill-defined memory and becomes non-deterministic. Pass such inputs through unchanged.
    return min(rgb.shape[:2]) >= 5


def inpaint_telea(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image)
    if not _inpaintable(rgb):
        return rgb
    return cv2.inpaint(rgb, _highlight_mask(rgb), 3, cv2.INPAINT_TELEA)


def inpaint_ns(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image)
    if not _inpaintable(rgb):
        return rgb
    return cv2.inpaint(rgb, _highlight_mask(rgb), 3, cv2.INPAINT_NS)


def flood_fill_center(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).copy()
    h, w = rgb.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    tol = int(params["tolerance"])
    cv2.floodFill(rgb, mask, (w // 2, h // 2), (255, 0, 0), (tol,) * 3, (tol,) * 3)
    return rgb


def connected_components_colormap(image: Image, params: Params, seed: int) -> Image:
    binimg = to_binary(image)
    count, labels = cv2.connectedComponents(binimg)
    if count <= 1:
        return as_rgb(image)
    hue = (labels * (179.0 / max(1, count - 1))).astype(np.uint8)
    sat = np.full_like(hue, 200)
    val = np.where(labels > 0, 255, 0).astype(np.uint8)
    hsv = np.stack([hue, sat, val], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def contours_draw(image: Image, params: Params, seed: int) -> Image:
    binimg = to_binary(image)
    contours, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = as_rgb(image).copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 1)
    return out


def distance_transform(image: Image, params: Params, seed: int) -> Image:
    binimg = to_binary(image)
    dist = cv2.distanceTransform(binimg, cv2.DIST_L2, 3)
    peak = float(dist.max())
    return to_u8(dist * (255.0 / peak) if peak > 0 else dist)


def threshold_otsu(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image)
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out  # binary mask


def threshold_adaptive_mean(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image)
    block = odd_ge1(max(3, int(params["blocksize"])))
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, int(params["c"])
    )


def palette_reduce_kmeans(image: Image, params: Params, seed: int) -> Image:
    """Reduce to `k` colours via k-means (platform-sensitive; seeded via cv2 RNG)."""
    rgb = as_rgb(image)
    samples = rgb.reshape(-1, 3).astype(np.float32)
    k = min(int(params["k"]), samples.shape[0])  # clamp for degenerate (e.g. 1-pixel) inputs
    cv2.setRNGSeed(int(seed))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # bestLabels=None is valid at runtime (output is allocated); the cv2 stub omits the overload.
    _, labels, centers = cv2.kmeans(  # type: ignore[call-overload]
        samples, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS
    )
    quantized = centers[labels.flatten()].reshape(rgb.shape)
    return to_u8(quantized)


def dither_floyd_steinberg(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image).astype(np.float32)
    levels = max(2, int(params["levels"]))
    h, w = gray.shape
    out = gray.copy()
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = round(old / 255.0 * (levels - 1)) / (levels - 1) * 255.0
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                out[y, x + 1] += err * 7.0 / 16.0
            if y + 1 < h:
                if x > 0:
                    out[y + 1, x - 1] += err * 3.0 / 16.0
                out[y + 1, x] += err * 5.0 / 16.0
                if x + 1 < w:
                    out[y + 1, x + 1] += err * 1.0 / 16.0
    return to_u8(out)


def pixel_sort_rows(image: Image, params: Params, seed: int) -> Image:
    """Sort bright spans of each row by luminance (deterministic)."""
    rgb = as_rgb(image).copy()
    lum = to_gray(image)
    threshold = int(params["threshold"])
    for y in range(rgb.shape[0]):
        idx = np.where(lum[y] > threshold)[0]
        if idx.size > 1:
            order = idx[np.argsort(lum[y, idx], kind="stable")]
            rgb[y, idx] = rgb[y, order]
    return rgb


def mosaic_pixelate(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image)
    h, w = rgb.shape[:2]
    b = max(1, int(params["block"]))
    small = cv2.resize(rgb, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def vignette(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).astype(np.float32)
    h, w = rgb.shape[:2]
    strength = float(params["strength"])
    ky = cv2.getGaussianKernel(h, h * strength)
    kx = cv2.getGaussianKernel(w, w * strength)
    mask = ky @ kx.T
    mask = mask / mask.max()
    return to_u8(rgb * mask[:, :, None])


def gradient_map_duotone(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image)
    dark = np.array([20, 20, 80], np.float32)
    light = np.array([240, 220, 120], np.float32)
    ramp = np.arange(256, dtype=np.float32)[:, None] / 255.0
    lut = to_u8(dark[None, :] * (1.0 - ramp) + light[None, :] * ramp)  # (256, 3)
    return np.ascontiguousarray(lut[gray])


def checkerboard_synthesize(image: Image, params: Params, seed: int) -> Image:
    h, w = image.shape[:2]
    n = max(1, int(params["squares"]))
    ys = (np.arange(h) // max(1, h // n))[:, None]
    xs = (np.arange(w) // max(1, w // n))[None, :]
    board = (((ys + xs) % 2) * 255).astype(np.uint8)
    return cv2.cvtColor(board, cv2.COLOR_GRAY2RGB)


def border_frame(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).copy()
    b, v = max(1, int(params["width"])), int(params["value"])
    rgb[:b, :] = v
    rgb[-b:, :] = v
    rgb[:, :b] = v
    rgb[:, -b:] = v
    return rgb


def overlay_grid(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).copy()
    s, v = max(1, int(params["spacing"])), int(params["value"])
    rgb[::s, :] = v
    rgb[:, ::s] = v
    return rgb


def sketch_pencil(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    return cv2.divide(gray, 255 - blur, scale=256)


def cartoonize(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image)
    color = cv2.bilateralFilter(rgb, 9, 150, 150)
    gray = to_gray(image)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def equalize_per_channel(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image)
    chans = [cv2.equalizeHist(rgb[:, :, i]) for i in range(3)]
    return np.ascontiguousarray(np.stack(chans, axis=-1))


def blur_background_composite(image: Image, params: Params, seed: int) -> Image:
    """Fake depth-of-field: sharp centre, blurred edges via a radial mask."""
    rgb = as_rgb(image).astype(np.float32)
    h, w = rgb.shape[:2]
    k = odd_ge1(int(params["ksize"]))
    blurred = cv2.GaussianBlur(rgb, (k, k), 0)
    yy, xx = np.ogrid[:h, :w]
    radial = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    mask = np.clip(1.0 - radial, 0.0, 1.0)[:, :, None]
    return to_u8(rgb * mask + blurred * (1.0 - mask))


def mask_to_rgba(image: Image, params: Params, seed: int) -> Image:
    """Emit RGBA with alpha = luma-threshold mask (exercises the alpha canonicalization path)."""
    rgb = as_rgb(image)
    gray = to_gray(image)
    alpha = ((gray > int(params["thresh"])).astype(np.uint8)) * 255
    return np.dstack([rgb, alpha])


def halftone_dots(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image)
    h, w = gray.shape
    cell = max(2, int(params["cell"]))
    out = np.full((h, w), 255, np.uint8)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            block = gray[y : y + cell, x : x + cell]
            intensity = 1.0 - float(block.mean()) / 255.0
            radius = int(intensity * cell / 2.0)
            if radius > 0:
                cv2.circle(out, (x + cell // 2, y + cell // 2), radius, 0, -1)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def value_noise_synthesize(image: Image, params: Params, seed: int) -> Image:
    """Seeded synthesis: upsampled low-resolution random noise at the input's size."""
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)
    s = max(1, int(params["scale"]))
    small = rng.integers(0, 256, size=(max(1, h // s), max(1, w // s), 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def swirl_distort(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image)
    h, w = rgb.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    strength = float(params["strength"])
    yy, xx = np.indices((h, w), dtype=np.float32)
    dx, dy = xx - cx, yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx) + strength * radius / max(h, w)
    map_x = (cx + radius * np.cos(theta)).astype(np.float32)
    map_y = (cy + radius * np.sin(theta)).astype(np.float32)
    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def build() -> list[Skill]:
    return [
        make_skill(
            "inpaint_telea_v1",
            "Inpaint (Telea)",
            "Inpaint saturated highlights with the Telea algorithm.",
            inpaint_telea,
            "reconstruction",
        ),
        make_skill(
            "inpaint_ns_v1",
            "Inpaint (Navier-Stokes)",
            "Inpaint saturated highlights with the Navier-Stokes algorithm.",
            inpaint_ns,
            "reconstruction",
        ),
        make_skill(
            "flood_fill_center_v1",
            "Flood fill (center)",
            "Flood-fill from the image centre within a tolerance.",
            flood_fill_center,
            "reconstruction",
            (param("tolerance", "int", 20, minimum=0, maximum=255),),
        ),
        make_skill(
            "connected_components_colormap_v1",
            "Connected components",
            "Label connected components and colour-map them.",
            connected_components_colormap,
            "segmentation",
        ),
        make_skill(
            "contours_draw_v1",
            "Draw contours",
            "Trace and draw external contours.",
            contours_draw,
            "segmentation",
        ),
        make_skill(
            "distance_transform_v1",
            "Distance transform",
            "Euclidean distance transform of the foreground.",
            distance_transform,
            "segmentation",
        ),
        make_skill(
            "threshold_otsu_v1",
            "Threshold (Otsu)",
            "Automatic global threshold via Otsu's method.",
            threshold_otsu,
            "segmentation",
        ),
        make_skill(
            "threshold_adaptive_mean_v1",
            "Threshold (adaptive)",
            "Adaptive mean thresholding.",
            threshold_adaptive_mean,
            "segmentation",
            (
                param("blocksize", "int", 11, minimum=3, maximum=99),
                param("c", "int", 2, minimum=-32, maximum=32),
            ),
        ),
        make_skill(
            "palette_reduce_kmeans_v1",
            "Palette reduce (k-means)",
            "Reduce to k colours via k-means clustering.",
            palette_reduce_kmeans,
            "quantize",
            (param("k", "int", 8, minimum=2, maximum=64),),
            platform_sensitive=True,
        ),
        make_skill(
            "dither_floyd_steinberg_v1",
            "Dither (Floyd-Steinberg)",
            "Error-diffusion dithering to N levels.",
            dither_floyd_steinberg,
            "quantize",
            (param("levels", "int", 2, minimum=2, maximum=16),),
        ),
        make_skill(
            "pixel_sort_rows_v1",
            "Pixel sort (rows)",
            "Sort bright spans of each row by luminance.",
            pixel_sort_rows,
            "stylize",
            (param("threshold", "int", 128, minimum=0, maximum=255),),
        ),
        make_skill(
            "mosaic_pixelate_v1",
            "Mosaic / pixelate",
            "Pixelate with square blocks.",
            mosaic_pixelate,
            "stylize",
            (param("block", "int", 8, minimum=1, maximum=128),),
        ),
        make_skill(
            "vignette_v1",
            "Vignette",
            "Darken toward the edges with a radial falloff.",
            vignette,
            "stylize",
            (param("strength", "float", 0.5, minimum=0.1, maximum=2.0),),
        ),
        make_skill(
            "gradient_map_duotone_v1",
            "Gradient map (duotone)",
            "Map luma through a two-colour gradient.",
            gradient_map_duotone,
            "stylize",
        ),
        make_skill(
            "checkerboard_synthesize_v1",
            "Checkerboard",
            "Synthesize a checkerboard at the input's size.",
            checkerboard_synthesize,
            "synthesis",
            (param("squares", "int", 8, minimum=1, maximum=64),),
        ),
        make_skill(
            "border_frame_v1",
            "Border frame",
            "Draw a solid frame around the image.",
            border_frame,
            "synthesis",
            (
                param("width", "int", 8, minimum=1, maximum=128),
                param("value", "int", 0, minimum=0, maximum=255),
            ),
        ),
        make_skill(
            "overlay_grid_v1",
            "Overlay grid",
            "Overlay a regular grid of lines.",
            overlay_grid,
            "synthesis",
            (
                param("spacing", "int", 16, minimum=2, maximum=128),
                param("value", "int", 0, minimum=0, maximum=255),
            ),
        ),
        make_skill(
            "sketch_pencil_v1",
            "Pencil sketch",
            "Convert to a pencil-sketch look.",
            sketch_pencil,
            "stylize",
        ),
        make_skill(
            "cartoonize_v1",
            "Cartoonize",
            "Bilateral colour + adaptive edges cartoon effect.",
            cartoonize,
            "stylize",
        ),
        make_skill(
            "equalize_per_channel_v1",
            "Equalize per channel",
            "Histogram-equalize each RGB channel independently.",
            equalize_per_channel,
            "reconstruction",
        ),
        make_skill(
            "blur_background_composite_v1",
            "Background blur composite",
            "Sharp centre composited over a blurred background.",
            blur_background_composite,
            "reconstruction",
            (param("ksize", "int", 15, minimum=1, maximum=51),),
        ),
        make_skill(
            "mask_to_rgba_v1",
            "Mask to RGBA",
            "Use a luma threshold as the alpha channel (RGBA output).",
            mask_to_rgba,
            "mask",
            (param("thresh", "int", 128, minimum=0, maximum=255),),
        ),
        make_skill(
            "halftone_dots_v1",
            "Halftone dots",
            "Render as a halftone dot pattern.",
            halftone_dots,
            "stylize",
            (param("cell", "int", 6, minimum=2, maximum=32),),
        ),
        make_skill(
            "value_noise_synthesize_v1",
            "Value noise",
            "Synthesize upsampled random value noise (seeded).",
            value_noise_synthesize,
            "synthesis",
            (param("scale", "int", 8, minimum=1, maximum=64),),
            seeded_stochastic=True,
        ),
        make_skill(
            "swirl_distort_v1",
            "Swirl distort",
            "Apply a radial swirl distortion.",
            swirl_distort,
            "stylize",
            (param("strength", "float", 5.0, minimum=-20.0, maximum=20.0),),
        ),
    ]

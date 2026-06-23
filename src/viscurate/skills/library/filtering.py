"""Signal / blur / edges / morphology skills (CLAUDE.md families 51–75).

Planted relations exercised later: ``blur_gaussian`` vs ``blur_box`` (the headline hard
negative — agree at small k, diverge at large k), the Sobel/Scharr/Prewitt/Canny/Laplacian
edge family (SEMANTIC neighbours), ``morphology_dilate`` ↔ ``morphology_erode``
(COMPLEMENTARY), and ``morphology_open``/``close`` built from them.
``low_pass_fft``/``high_pass_fft`` are flagged **precision-sensitive** (§1.4); the noise
skills are **seeded-stochastic** determinism probes.
"""

from __future__ import annotations

import cv2
import numpy as np

from viscurate.skills.library._build import make_skill, param
from viscurate.skills.library._ops import as_rgb, odd_ge1, to_gray, to_u8
from viscurate.skills.model import Image, Params, Skill


def blur_gaussian(image: Image, params: Params, seed: int) -> Image:
    k = odd_ge1(int(params["ksize"]))
    return cv2.GaussianBlur(image, (k, k), float(params["sigma"]))


def blur_box(image: Image, params: Params, seed: int) -> Image:
    k = odd_ge1(int(params["ksize"]))
    return cv2.blur(image, (k, k))


def blur_median(image: Image, params: Params, seed: int) -> Image:
    return cv2.medianBlur(as_rgb(image), odd_ge1(int(params["ksize"])))


def blur_bilateral(image: Image, params: Params, seed: int) -> Image:
    sigma = float(params["sigma"])
    return cv2.bilateralFilter(as_rgb(image), int(params["d"]), sigma, sigma)


def blur_motion(image: Image, params: Params, seed: int) -> Image:
    length = max(1, int(params["length"]))
    kernel: Image = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D(
        (length / 2 - 0.5, length / 2 - 0.5), float(params["angle"]), 1.0
    )
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    total = kernel.sum()
    if total > 0:
        kernel = kernel / total
    return cv2.filter2D(as_rgb(image), -1, kernel)


def sharpen_unsharp(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).astype(np.float32)
    k = odd_ge1(int(params["ksize"]))
    blurred = cv2.GaussianBlur(rgb, (k, k), 0)
    amount = float(params["amount"])
    return to_u8(rgb + amount * (rgb - blurred))


def sharpen_laplacian_kernel(image: Image, params: Params, seed: int) -> Image:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(as_rgb(image), -1, kernel)


def high_pass_spatial(image: Image, params: Params, seed: int) -> Image:
    rgb = as_rgb(image).astype(np.float32)
    k = odd_ge1(int(params["ksize"]))
    low = cv2.GaussianBlur(rgb, (k, k), 0)
    return to_u8(rgb - low + 128.0)


def emboss(image: Image, params: Params, seed: int) -> Image:
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    gray = to_gray(image).astype(np.float32)
    return to_u8(cv2.filter2D(gray, -1, kernel) + 128.0)


def edges_sobel(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return to_u8(np.sqrt(gx * gx + gy * gy))


def edges_scharr(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return to_u8(np.sqrt(gx * gx + gy * gy))


def edges_prewitt(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image).astype(np.float32)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    return to_u8(np.sqrt(gx * gx + gy * gy))


def edges_laplacian(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image)
    return to_u8(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)))


def edges_canny(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image)
    return cv2.Canny(gray, int(params["low"]), int(params["high"]))  # uint8 {0,255}


def difference_of_gaussians(image: Image, params: Params, seed: int) -> Image:
    gray = to_gray(image).astype(np.float32)
    b1 = cv2.GaussianBlur(gray, (0, 0), float(params["sigma1"]))
    b2 = cv2.GaussianBlur(gray, (0, 0), float(params["sigma2"]))
    return to_u8(np.abs(b1 - b2))


def gabor_filter(image: Image, params: Params, seed: int) -> Image:
    k = odd_ge1(int(params["ksize"]))
    kernel = cv2.getGaborKernel(
        (k, k),
        float(params["sigma"]),
        np.deg2rad(float(params["theta"])),
        float(params["lambd"]),
        0.5,
        0.0,
        ktype=cv2.CV_32F,
    )
    gray = to_gray(image).astype(np.float32)
    return to_u8(cv2.filter2D(gray, -1, kernel))


def _fft_filter(image: Image, cutoff: float, *, high: bool) -> Image:
    gray = to_gray(image).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = max(1, int(cutoff * min(h, w)))
    mask: Image = np.zeros((h, w), np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1)
    if high:
        mask = 1.0 - mask
    out = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
    return to_u8(np.abs(out))


def low_pass_fft(image: Image, params: Params, seed: int) -> Image:
    """Keep low frequencies (precision-sensitive: FFT round-trip)."""
    return _fft_filter(image, float(params["cutoff"]), high=False)


def high_pass_fft(image: Image, params: Params, seed: int) -> Image:
    """Keep high frequencies (precision-sensitive: FFT round-trip)."""
    return _fft_filter(image, float(params["cutoff"]), high=True)


def morphology_dilate(image: Image, params: Params, seed: int) -> Image:
    k = odd_ge1(int(params["ksize"]))
    return cv2.dilate(as_rgb(image), np.ones((k, k), np.uint8), iterations=1)


def morphology_erode(image: Image, params: Params, seed: int) -> Image:
    k = odd_ge1(int(params["ksize"]))
    return cv2.erode(as_rgb(image), np.ones((k, k), np.uint8), iterations=1)


def morphology_open(image: Image, params: Params, seed: int) -> Image:
    k = odd_ge1(int(params["ksize"]))
    return cv2.morphologyEx(as_rgb(image), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))


def morphology_close(image: Image, params: Params, seed: int) -> Image:
    k = odd_ge1(int(params["ksize"]))
    return cv2.morphologyEx(as_rgb(image), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))


def denoise_nlmeans(image: Image, params: Params, seed: int) -> Image:
    s = float(params["strength"])
    return cv2.fastNlMeansDenoisingColored(as_rgb(image), None, s, s, 7, 21)


def add_gaussian_noise(image: Image, params: Params, seed: int) -> Image:
    """Seeded-stochastic: deterministic at a fixed seed, so comparable at matched seeds."""
    rgb = as_rgb(image).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(params["sigma"]), size=rgb.shape).astype(np.float32)
    return to_u8(rgb + noise)


def add_salt_pepper_noise(image: Image, params: Params, seed: int) -> Image:
    """Seeded-stochastic salt-and-pepper corruption."""
    rgb = as_rgb(image).copy()
    rng = np.random.default_rng(seed)
    amount = float(params["amount"])
    roll = rng.random(rgb.shape[:2])
    rgb[roll < amount / 2.0] = 0
    rgb[roll > 1.0 - amount / 2.0] = 255
    return rgb


def build() -> list[Skill]:
    return [
        make_skill(
            "blur_gaussian_v1",
            "Gaussian blur",
            "Convolve with a Gaussian kernel.",
            blur_gaussian,
            "blur",
            (
                param("ksize", "int", 5, minimum=1, maximum=31),
                param("sigma", "float", 0.0, minimum=0.0, maximum=20.0),
            ),
        ),
        make_skill(
            "blur_box_v1",
            "Box blur",
            "Convolve with a uniform box kernel.",
            blur_box,
            "blur",
            (param("ksize", "int", 5, minimum=1, maximum=31),),
        ),
        make_skill(
            "blur_median_v1",
            "Median blur",
            "Replace each pixel with the neighbourhood median.",
            blur_median,
            "blur",
            (param("ksize", "int", 5, minimum=1, maximum=31),),
        ),
        make_skill(
            "blur_bilateral_v1",
            "Bilateral blur",
            "Edge-preserving bilateral smoothing.",
            blur_bilateral,
            "blur",
            (
                param("d", "int", 9, minimum=1, maximum=25),
                param("sigma", "float", 75.0, minimum=1.0, maximum=200.0),
            ),
        ),
        make_skill(
            "blur_motion_v1",
            "Motion blur",
            "Directional motion blur.",
            blur_motion,
            "blur",
            (
                param("length", "int", 9, minimum=1, maximum=51),
                param("angle", "float", 0.0, minimum=-180.0, maximum=180.0),
            ),
        ),
        make_skill(
            "sharpen_unsharp_v1",
            "Unsharp mask",
            "Sharpen via the unsharp-mask method.",
            sharpen_unsharp,
            "sharpen",
            (
                param("ksize", "int", 5, minimum=1, maximum=31),
                param("amount", "float", 1.0, minimum=0.0, maximum=5.0),
            ),
        ),
        make_skill(
            "sharpen_laplacian_kernel_v1",
            "Laplacian sharpen",
            "Sharpen with a fixed Laplacian kernel.",
            sharpen_laplacian_kernel,
            "sharpen",
        ),
        make_skill(
            "high_pass_spatial_v1",
            "High-pass (spatial)",
            "Spatial high-pass (image minus Gaussian low-pass).",
            high_pass_spatial,
            "frequency",
            (param("ksize", "int", 9, minimum=1, maximum=31),),
        ),
        make_skill("emboss_v1", "Emboss", "Directional emboss relief.", emboss, "stylize"),
        make_skill(
            "edges_sobel_v1",
            "Sobel edges",
            "Gradient magnitude via Sobel operators.",
            edges_sobel,
            "edges",
        ),
        make_skill(
            "edges_scharr_v1",
            "Scharr edges",
            "Gradient magnitude via Scharr operators.",
            edges_scharr,
            "edges",
        ),
        make_skill(
            "edges_prewitt_v1",
            "Prewitt edges",
            "Gradient magnitude via Prewitt operators.",
            edges_prewitt,
            "edges",
        ),
        make_skill(
            "edges_laplacian_v1",
            "Laplacian edges",
            "Edge response via the Laplacian operator.",
            edges_laplacian,
            "edges",
        ),
        make_skill(
            "edges_canny_v1",
            "Canny edges",
            "Canny edge detector (binary edge map).",
            edges_canny,
            "edges",
            (
                param("low", "int", 100, minimum=0, maximum=500),
                param("high", "int", 200, minimum=0, maximum=500),
            ),
        ),
        make_skill(
            "difference_of_gaussians_v1",
            "Difference of Gaussians",
            "Band-pass via a difference of two Gaussians.",
            difference_of_gaussians,
            "edges",
            (
                param("sigma1", "float", 1.0, minimum=0.1, maximum=20.0),
                param("sigma2", "float", 2.0, minimum=0.1, maximum=20.0),
            ),
        ),
        make_skill(
            "gabor_filter_v1",
            "Gabor filter",
            "Oriented Gabor texture response.",
            gabor_filter,
            "texture",
            (
                param("ksize", "int", 15, minimum=3, maximum=51),
                param("sigma", "float", 4.0, minimum=0.5, maximum=20.0),
                param("theta", "float", 0.0, minimum=0.0, maximum=180.0),
                param("lambd", "float", 10.0, minimum=2.0, maximum=40.0),
            ),
        ),
        make_skill(
            "low_pass_fft_v1",
            "Low-pass (FFT)",
            "Keep low frequencies via an FFT round-trip.",
            low_pass_fft,
            "frequency",
            (param("cutoff", "float", 0.25, minimum=0.01, maximum=0.5),),
            precision_sensitive=True,
        ),
        make_skill(
            "high_pass_fft_v1",
            "High-pass (FFT)",
            "Keep high frequencies via an FFT round-trip.",
            high_pass_fft,
            "frequency",
            (param("cutoff", "float", 0.25, minimum=0.01, maximum=0.5),),
            precision_sensitive=True,
        ),
        make_skill(
            "morphology_dilate_v1",
            "Dilate",
            "Grayscale/colour morphological dilation.",
            morphology_dilate,
            "morphology",
            (param("ksize", "int", 3, minimum=1, maximum=31),),
        ),
        make_skill(
            "morphology_erode_v1",
            "Erode",
            "Grayscale/colour morphological erosion.",
            morphology_erode,
            "morphology",
            (param("ksize", "int", 3, minimum=1, maximum=31),),
        ),
        make_skill(
            "morphology_open_v1",
            "Open",
            "Erosion followed by dilation (removes speckle).",
            morphology_open,
            "morphology",
            (param("ksize", "int", 3, minimum=1, maximum=31),),
        ),
        make_skill(
            "morphology_close_v1",
            "Close",
            "Dilation followed by erosion (fills holes).",
            morphology_close,
            "morphology",
            (param("ksize", "int", 3, minimum=1, maximum=31),),
        ),
        make_skill(
            "denoise_nlmeans_v1",
            "Denoise (NL-means)",
            "Non-local-means colour denoising.",
            denoise_nlmeans,
            "denoise",
            (param("strength", "float", 10.0, minimum=1.0, maximum=50.0),),
        ),
        make_skill(
            "add_gaussian_noise_v1",
            "Add Gaussian noise",
            "Add zero-mean Gaussian noise (seeded).",
            add_gaussian_noise,
            "noise",
            (param("sigma", "float", 10.0, minimum=0.0, maximum=128.0),),
            seeded_stochastic=True,
        ),
        make_skill(
            "add_salt_pepper_noise_v1",
            "Add salt & pepper noise",
            "Add salt-and-pepper noise (seeded).",
            add_salt_pepper_noise,
            "noise",
            (param("amount", "float", 0.05, minimum=0.0, maximum=1.0),),
            seeded_stochastic=True,
        ),
    ]

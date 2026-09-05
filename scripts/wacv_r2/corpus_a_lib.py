"""A1 Corpus A — adapters and Skill construction for cross-library functional cores.

Each entry in ``corpus_a_alignment.yaml`` names a deterministic functional core plus the
representation it expects. This module turns those declarations into repo-native
:class:`viscurate.skills.model.Skill` objects so the UNMODIFIED verifier
(:func:`viscurate.equivalence.taxonomy.classify`) decides every pair — no parallel comparison
path, and no way for a cross-library adapter to bypass the text-blindness contract, since the
comparator is still handed a ``ComparatorView``.

CONVERSION LOGGING. Every adapter records the exact conversion chain it applied. A dtype/layout
bug is the most likely source of a false divergence finding (A1/01 §5 step 3), so the chain is
written into the run artifact per implementation and the self-pair gate is what proves the chain
round-trips: an implementation whose adapter loses information cannot certify EXACT against
itself, so the gate fails loudly instead of producing a plausible wrong number.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# albucore (albumentations' kernel library) keys MAX_VALUES_BY_DTYPE by numpy scalar types that
# numpy>=2 no longer registers, so importing albumentations raises KeyError(np.uint32). The shim
# adds the missing keys with their true maxima. Recorded in the environment manifest because it is
# a deviation from a stock install.
import albucore
import cv2
import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageFilter
import PIL.ImageOps
import scipy.ndimage
import skimage.color
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.transform
import skimage.util
import torch
import torchvision.transforms
import torchvision.transforms.v2.functional

for _t in (np.uint8, np.uint16, np.uint32, np.int32, np.float32, np.float64):
    if _t not in albucore.MAX_VALUES_BY_DTYPE:
        albucore.MAX_VALUES_BY_DTYPE[_t] = albucore.MAX_VALUES_BY_DTYPE.get(
            np.dtype(_t),
            float(np.iinfo(_t).max) if np.issubdtype(_t, np.integer) else 1.0,
        )

import albumentations  # noqa: E402
import albumentations.augmentations.blur.functional  # noqa: E402
import albumentations.augmentations.geometric.functional  # noqa: E402
import albumentations.augmentations.pixel.functional  # noqa: E402
import kornia  # noqa: E402

# Fork-pair libraries for the pilgram/pilgram2 study. Installed into a side directory
# (`.corpusb_libs`, put on PYTHONPATH by the runner) rather than into the shared env, so that a
# transitive dependency upgrade could not disturb verification jobs that were already running.
# Guarded because a stock checkout will not have them.
try:  # pragma: no cover - optional corpus
    import pilgram
    import pilgram2
except ImportError:  # pragma: no cover
    pilgram = None  # type: ignore[assignment]
    pilgram2 = None  # type: ignore[assignment]

ALBUCORE_SHIM = "albucore.MAX_VALUES_BY_DTYPE += {uint8,uint16,uint32,int32,float32,float64}"

Image = np.ndarray


# ==============================================================================================
# Adapters: uint8 HWC RGB  <->  the representation each library's functional core expects.
# ==============================================================================================


def _to_u8_hwc_rgb(img: Image) -> tuple[Any, list[str]]:
    return img, ["identity"]


def _from_u8_hwc_rgb(out: Any) -> Image:
    return np.asarray(out)


def _to_f32_hwc_01(img: Image) -> tuple[Any, list[str]]:
    return img.astype(np.float32) / 255.0, ["uint8->float32", "[0,255]->[0,1]"]


def _from_f32_hwc_01(out: Any) -> Image:
    arr = np.asarray(out, dtype=np.float32)
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _to_pil_rgb(img: Image) -> tuple[Any, list[str]]:
    return PIL.Image.fromarray(img, mode="RGB"), ["ndarray->PIL.Image(RGB)"]


def _from_pil_rgb(out: Any) -> Image:
    if isinstance(out, PIL.Image.Image):
        return np.asarray(out.convert("RGB"), dtype=np.uint8)
    return np.asarray(out)


def _to_torch_bchw_01(img: Image) -> tuple[Any, list[str]]:
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
    return t.to(torch.float32) / 255.0, ["HWC->BCHW", "uint8->float32", "[0,255]->[0,1]"]


def _from_torch_bchw_01(out: Any) -> Image:
    t = out.detach().to(torch.float32).cpu()
    if t.dim() == 4:
        t = t[0]
    arr = t.permute(1, 2, 0).numpy()
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _to_torch_bchw_u8(img: Image) -> tuple[Any, list[str]]:
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
    return t.contiguous(), ["HWC->BCHW", "uint8 kept"]


def _from_torch_bchw_u8(out: Any) -> Image:
    t = out.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.dtype != torch.uint8:
        return np.clip(t.permute(1, 2, 0).to(torch.float32).numpy(), 0, 255).astype(np.uint8)
    return t.permute(1, 2, 0).numpy().astype(np.uint8)


ADAPTERS: dict[str, tuple[Callable[[Image], tuple[Any, list[str]]], Callable[[Any], Image]]] = {
    "u8_hwc_rgb": (_to_u8_hwc_rgb, _from_u8_hwc_rgb),
    "f32_hwc_01": (_to_f32_hwc_01, _from_f32_hwc_01),
    "pil_rgb": (_to_pil_rgb, _from_pil_rgb),
    "torch_bchw_01": (_to_torch_bchw_01, _from_torch_bchw_01),
    "torch_bchw_u8": (_to_torch_bchw_u8, _from_torch_bchw_u8),
}


# ==============================================================================================
# Helpers referenced by name from the alignment file's `call` expressions.
# ==============================================================================================


def _sobel_mag_cv2(img: Image, ksize: int) -> Image:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(dx, dy)
    return cv2.cvtColor(cv2.convertScaleAbs(mag), cv2.COLOR_GRAY2RGB)


CALL_GLOBALS: dict[str, Any] = {
    "np": np,
    "cv2": cv2,
    "torch": torch,
    "kornia": kornia,
    "skimage": skimage,
    "scipy": scipy,
    "PIL": PIL,
    "torchvision": torchvision,
    "albumentations": albumentations,
    "_sobel_mag_cv2": _sobel_mag_cv2,
}

if pilgram is not None:
    CALL_GLOBALS["pilgram"] = pilgram
if pilgram2 is not None:
    CALL_GLOBALS["pilgram2"] = pilgram2

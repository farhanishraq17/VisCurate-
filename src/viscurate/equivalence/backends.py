"""Perceptual / semantic distance backends (CLAUDE.md §3, §3.5.1).

The comparators need three views of an output pair:

* **LPIPS (AlexNet)** — the PERCEPTUAL metric (§3.5.4).
* **DINO ViT-B/16** — the SEMANTIC feature space; CLIP ViT-B/32 is an optional, more
  conservative second view (take the larger distance, §3.5.4).
* **SSIM** — a cheap structural cross-check / floor under PERCEPTUAL (§3.1).

Two design rules hold here:

* **Lazy, optional ML.** ``torch`` / ``lpips`` / ``timm`` / ``open_clip`` are imported only
  inside the backend constructors, so this module imports cleanly without the ``[ml]`` extra.
  Construct a backend only when you have the extra installed.
* **One model at a time.** Each backend owns one model and frees it on :meth:`close` (it is a
  context manager). The 6 GB-GPU budget (CLAUDE.md D5) is met by loading the perceptual model
  for the PERCEPTUAL stage, freeing it, then loading the semantic model — never both at once.

All backends operate on the **canonical RGB view**: ``float32`` arrays of shape ``(H, W, 3)``
in ``[0, 1]`` (see :mod:`viscurate.skills.canonicalize`). They never see text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch

__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ClipBackend",
    "DinoBackend",
    "LpipsBackend",
    "PerceptualBackend",
    "SemanticBackend",
    "cosine_distance",
    "ssim_distance",
]

NDArrayF = npt.NDArray[np.float32]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@runtime_checkable
class PerceptualBackend(Protocol):
    """A perceptual distance between two canonical RGB outputs (lower = more similar)."""

    name: str

    def distance(self, a: NDArrayF, b: NDArrayF) -> float: ...


@runtime_checkable
class SemanticBackend(Protocol):
    """Maps canonical RGB outputs to L2-normalized feature vectors (cosine compared)."""

    name: str

    def features(self, imgs: Sequence[NDArrayF]) -> NDArrayF:
        """Return an ``(N, D)`` float32 array of L2-normalized features."""
        ...


def cosine_distance(u: NDArrayF, v: NDArrayF) -> float:
    """``1 - cosine_similarity`` for two (already-normalized) vectors, clipped to [0, 2]."""
    uu = np.asarray(u, dtype=np.float64).ravel()
    vv = np.asarray(v, dtype=np.float64).ravel()
    nu = float(np.linalg.norm(uu))
    nv = float(np.linalg.norm(vv))
    if nu == 0.0 or nv == 0.0:
        return 1.0
    sim = float(np.dot(uu, vv) / (nu * nv))
    return float(np.clip(1.0 - sim, 0.0, 2.0))


def ssim_distance(a: NDArrayF, b: NDArrayF) -> float:
    """``1 - SSIM`` over the canonical RGB view (structural floor under PERCEPTUAL).

    Returns ``inf`` on a shape mismatch (the shape gate). Lazy-imports scikit-image so this
    module stays importable without the ``[ml]`` extra.
    """
    if a.shape != b.shape:
        return float("inf")
    from skimage.metrics import structural_similarity

    win = min(7, a.shape[0], a.shape[1])
    if win < 3:  # too small for a structural window — fall back to "identical iff equal"
        return 0.0 if bool(np.array_equal(a, b)) else 1.0
    if win % 2 == 0:
        win -= 1
    score = structural_similarity(a, b, channel_axis=2, data_range=1.0, win_size=win)
    return float(np.clip(1.0 - float(score), 0.0, 2.0))


# --------------------------------------------------------------------------------------------
# Real backends (require the [ml] extra). torch & friends import lazily in __init__.
# --------------------------------------------------------------------------------------------


def _as_nchw(imgs: Sequence[NDArrayF], device: Any) -> torch.Tensor:
    """Stack (H, W, 3) float[0,1] arrays into an (N, 3, H, W) tensor on ``device``."""
    import torch

    arr = np.stack([np.ascontiguousarray(im, dtype=np.float32) for im in imgs], axis=0)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    return t.to(device)


def _resize_normalize(
    x: torch.Tensor, size: int, mean: tuple[float, float, float], std: tuple[float, float, float]
) -> torch.Tensor:
    """Resize an (N, 3, H, W) tensor to ``size`` and apply per-channel mean/std normalization.

    DINO/CLIP are resize-tolerant (CLAUDE.md §3.5.4); bilinear keeps it deterministic.
    """
    import torch
    import torch.nn.functional as F

    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


class LpipsBackend:
    """LPIPS (AlexNet) perceptual distance (CLAUDE.md §3.5.4)."""

    def __init__(self, net: str = "alex", device: str = "cpu") -> None:
        import lpips
        import torch

        self.name = f"lpips-{net}"
        self._device = torch.device(device)
        self._model: Any = lpips.LPIPS(net=net, verbose=False).to(self._device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    def distance(self, a: NDArrayF, b: NDArrayF) -> float:
        import torch

        if a.shape != b.shape:
            return float("inf")
        with torch.no_grad():
            ta = _as_nchw([a], self._device) * 2.0 - 1.0  # LPIPS expects [-1, 1]
            tb = _as_nchw([b], self._device) * 2.0 - 1.0
            if min(ta.shape[-2:]) < 64:
                import torch.nn.functional as F

                ta = F.interpolate(ta, size=(64, 64), mode="bilinear", align_corners=False)
                tb = F.interpolate(tb, size=(64, 64), mode="bilinear", align_corners=False)
            d = self._model(ta, tb)
        return float(d.reshape(-1)[0].item())

    def close(self) -> None:
        self._model = None

    def __enter__(self) -> LpipsBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _TimmFeatureBackend:
    """Shared machinery for timm feature extractors (DINO)."""

    def __init__(self, model_name: str, size: int, device: str) -> None:
        import timm
        import torch

        self.name = model_name
        self._size = size
        self._device = torch.device(device)
        self._model: Any = (
            timm.create_model(model_name, pretrained=True, num_classes=0).to(self._device).eval()
        )
        for p in self._model.parameters():
            p.requires_grad_(False)

    def features(self, imgs: Sequence[NDArrayF]) -> NDArrayF:
        import torch

        if not imgs:
            return np.zeros((0, 0), dtype=np.float32)
        out: list[NDArrayF] = []
        batch = 1
        with torch.no_grad():
            for i in range(0, len(imgs), batch):
                chunk = imgs[i : i + batch]
                x = _resize_normalize(
                    _as_nchw(chunk, self._device), self._size, IMAGENET_MEAN, IMAGENET_STD
                )
                f = self._model(x)
                f = torch.nn.functional.normalize(f, dim=1)
                out.append(np.asarray(f.cpu().numpy(), dtype=np.float32))
        return np.concatenate(out, axis=0)

    def close(self) -> None:
        self._model = None

    def __enter__(self) -> _TimmFeatureBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DinoBackend(_TimmFeatureBackend):
    """DINO ViT-B/16 self-supervised features (CLAUDE.md §3.5.4 SEMANTIC space)."""

    def __init__(self, model_name: str = "vit_base_patch16_224.dino", device: str = "cpu") -> None:
        super().__init__(model_name, size=224, device=device)


class ClipBackend:
    """CLIP ViT-B/32 image features — the optional, more conservative semantic view."""

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",  # matches the QuickGELU 'openai' weights
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        import open_clip
        import torch

        self.name = f"clip-{model_name}-{pretrained}"
        self._device = torch.device(device)
        self._model: Any = open_clip.create_model(model_name, pretrained=pretrained)
        self._model = self._model.to(self._device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    def features(self, imgs: Sequence[NDArrayF]) -> NDArrayF:
        import torch

        if not imgs:
            return np.zeros((0, 0), dtype=np.float32)
        out: list[NDArrayF] = []
        batch = 1
        with torch.no_grad():
            for i in range(0, len(imgs), batch):
                chunk = imgs[i : i + batch]
                x = _resize_normalize(_as_nchw(chunk, self._device), 224, CLIP_MEAN, CLIP_STD)
                f = self._model.encode_image(x)
                f = torch.nn.functional.normalize(f, dim=1)
                out.append(np.asarray(f.cpu().numpy(), dtype=np.float32))
        return np.concatenate(out, axis=0)

    def close(self) -> None:
        self._model = None

    def __enter__(self) -> ClipBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

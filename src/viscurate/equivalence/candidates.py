"""Output-based candidate generation (CLAUDE.md §3.5.6).

Full verification over the whole battery is too expensive for all ``O(N²)`` pairs. The naive
fix — "only verify pairs whose *descriptions* embed closely" — reintroduces the exact text bias
the project attacks and would miss the different-description/same-output redundancy that is half
the contribution. So candidates are proposed from **output fingerprints**, never text:

* a cheap **perceptual hash** (average-hash bits) plus a **mean DINO feature**, both computed on
  a small fixed *screening* sub-battery;
* nearest-neighbour over fingerprints proposes candidate pairs;
* **same-family pairs and the engineered hard negatives are always included**, so boundary
  cases (``blur_gaussian`` vs ``blur_box`` …) are never skipped.

Because the fingerprint is output-based, two skills with unrelated descriptions but identical
behaviour still collide and get verified — the redundancy text-based pruning misses.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from viscurate.equivalence.backends import SemanticBackend, cosine_distance
from viscurate.equivalence.compare import OutputProvider
from viscurate.skills.canonicalize import Canonical
from viscurate.skills.model import ComparatorView

__all__ = [
    "ENGINEERED_HARD_NEGATIVES",
    "candidate_pairs",
    "compute_fingerprints",
    "normalize_pair",
]

NDArrayF = npt.NDArray[np.float32]

# Planted boundary cases (CLAUDE.md §2.3, §3.5.4) that must never be pruned — keyed by id, not
# description, so this stays inside the output-grounded path.
ENGINEERED_HARD_NEGATIVES: tuple[tuple[str, str], ...] = (
    ("blur_gaussian_v1", "blur_box_v1"),
    ("resize_nearest_v1", "resize_bilinear_v1"),
    ("resize_bilinear_v1", "resize_bicubic_v1"),
    ("pad_reflect_v1", "pad_replicate_v1"),
    ("posterize_v1", "quantize_uniform_v1"),
)


def normalize_pair(a: str, b: str) -> tuple[str, str]:
    """An unordered pair as a sorted tuple, so ``(A, B)`` and ``(B, A)`` are one candidate."""
    return (a, b) if a <= b else (b, a)


def _ahash(c: Canonical, side: int = 8) -> NDArrayF:
    """Average-hash bits of a canonical RGB output (downsampled grayscale ≷ its mean)."""
    g = c.rgb.mean(axis=2)  # (H, W) grayscale
    h, w = g.shape
    # block-mean downsample to side×side (deterministic, no interpolation library needed)
    ys = np.linspace(0, h, side + 1).astype(int)
    xs = np.linspace(0, w, side + 1).astype(int)
    small = np.empty((side, side), dtype=np.float32)
    for i in range(side):
        for j in range(side):
            block = g[ys[i] : max(ys[i] + 1, ys[i + 1]), xs[j] : max(xs[j] + 1, xs[j + 1])]
            small[i, j] = float(block.mean()) if block.size else 0.0
    bits = (small >= small.mean()).astype(np.float32)
    return np.ascontiguousarray(bits.reshape(-1), dtype=np.float32)


def compute_fingerprints(
    views: Sequence[ComparatorView],
    provider: OutputProvider,
    semantic: SemanticBackend,
    *,
    screening_ids: Sequence[str],
    seed: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, NDArrayF]:
    """One L2-normalized output fingerprint per skill: [perceptual-hash ‖ mean DINO feature]."""
    fps: dict[str, NDArrayF] = {}
    total = len(views)
    for idx, v in enumerate(views, start=1):
        out = provider.outputs(v.id, seed=seed)
        ids = [p for p in screening_ids if p in out.canon]
        if not ids:
            ids = list(out.probe_ids[: len(screening_ids) or 8])
        hashes = (
            np.concatenate([_ahash(out.canon[p]) for p in ids]) if ids else np.zeros(64, np.float32)
        )
        feats = semantic.features([out.canon[p].rgb for p in ids])
        mean_feat = feats.mean(axis=0) if feats.size else np.zeros(1, np.float32)
        vec = np.concatenate([_unit(hashes), _unit(np.asarray(mean_feat, dtype=np.float32))])
        fps[v.id] = _unit(vec)
        if progress is not None:
            progress(idx, total, v.id)
    return fps


def _unit(x: NDArrayF) -> NDArrayF:
    n = float(np.linalg.norm(x))
    return (x / n).astype(np.float32) if n > 0 else x.astype(np.float32)


def candidate_pairs(
    views: Sequence[ComparatorView],
    fingerprints: dict[str, NDArrayF],
    *,
    k: int = 5,
    max_distance: float = 0.5,
    include_same_family: bool = True,
    hard_negatives: Sequence[tuple[str, str]] = ENGINEERED_HARD_NEGATIVES,
) -> set[tuple[str, str]]:
    """Propose candidate pairs by fingerprint NN ∪ same-family ∪ engineered hard negatives."""
    id_set = {v.id for v in views}
    pairs: set[tuple[str, str]] = set()

    for a in views:
        ranked = sorted(
            (cosine_distance(fingerprints[a.id], fingerprints[b.id]), b.id)
            for b in views
            if b.id != a.id
        )
        for dist, bid in ranked[:k]:
            if dist <= max_distance:
                pairs.add(normalize_pair(a.id, bid))

    if include_same_family:
        by_family: dict[str, list[str]] = defaultdict(list)
        for v in views:
            by_family[v.family].append(v.id)
        for members in by_family.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add(normalize_pair(members[i], members[j]))

    for a_id, b_id in hard_negatives:
        if a_id in id_set and b_id in id_set:
            pairs.add(normalize_pair(a_id, b_id))

    return pairs

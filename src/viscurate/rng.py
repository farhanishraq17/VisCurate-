"""Explicit, reproducible randomness — no global RNG state.

Determinism given ``(image, params, seed)`` is a hard requirement (CLAUDE.md §1.4): it is
what makes "same output" decidable. We therefore never call :func:`numpy.random.seed`
or rely on a process-global generator. Instead, every component derives a *child* seed
from a single root seed plus a list of string labels, and asks for its own
:class:`numpy.random.Generator`.

Seed derivation is a pure function of ``(root_seed, labels)`` via BLAKE2b, so the same
labels always yield the same stream regardless of call order or interleaving — the
property that lets us compare seeded-stochastic skills "at matched seeds".
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

__all__ = ["SeedManager", "derive_seed", "seed_global_libraries"]

_MASK64 = (1 << 64) - 1


def derive_seed(root_seed: int, *labels: str | int) -> int:
    """Deterministically derive a 64-bit child seed from a root seed and labels.

    Pure and order-independent across *different* label tuples: ``derive_seed(0, "a")``
    and ``derive_seed(0, "b")`` are independent, while ``derive_seed(0, "a")`` always
    returns the same value. Uses BLAKE2b so the mapping is stable across platforms and
    Python versions (unlike :func:`hash`, which is salted per-process).
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(
        int(root_seed).to_bytes(8, "little", signed=False)
        if root_seed >= 0
        else (root_seed & _MASK64).to_bytes(8, "little")
    )
    for label in labels:
        h.update(b"\x1f")  # unit separator so ("a","b") != ("ab",)
        h.update(str(label).encode("utf-8"))
    return int.from_bytes(h.digest(), "little")


class SeedManager:
    """Hands out per-component generators derived from one root seed.

    Example::

        sm = SeedManager(root_seed=1234)
        rng = sm.generator("probe", "noise", probe_id)   # np.random.Generator
        noise = rng.normal(size=(256, 256, 3))
    """

    __slots__ = ("_root",)

    def __init__(self, root_seed: int) -> None:
        self._root = int(root_seed)

    @property
    def root_seed(self) -> int:
        return self._root

    def child_seed(self, *labels: str | int) -> int:
        """Return the derived 64-bit seed for ``labels`` (no generator)."""
        return derive_seed(self._root, *labels)

    def generator(self, *labels: str | int) -> np.random.Generator:
        """Return a fresh :class:`numpy.random.Generator` seeded for ``labels``."""
        return np.random.default_rng(self.child_seed(*labels))

    def spawn(self, *labels: str | int) -> SeedManager:
        """Return a child manager whose root is this manager's derived seed for ``labels``.

        Lets a subsystem be handed a manager it can sub-divide without knowing the
        global root, while staying fully deterministic.
        """
        return SeedManager(self.child_seed(*labels))


def seed_global_libraries(seed: int, *, torch_too: bool = True) -> None:
    """Last-resort seeding of library-global RNGs (stdlib ``random``, NumPy legacy, torch).

    Prefer :class:`SeedManager` generators. This exists only for third-party code that
    reads a process-global RNG and offers no generator argument. ``torch`` is seeded only
    if installed, keeping this module import-light.
    """
    import random as _random

    _random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)

    if torch_too:
        try:
            import torch  # local import: keep rng.py ML-free unless asked
        except ImportError:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def labels_signature(labels: Sequence[str | int]) -> str:
    """Human-readable signature of a label tuple, for logging/manifests."""
    return "/".join(str(x) for x in labels)

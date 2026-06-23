from __future__ import annotations

import numpy as np

from viscurate.rng import SeedManager, derive_seed


def test_derive_seed_is_pure_and_stable() -> None:
    # Stable across calls and (by construction) across processes/platforms.
    assert derive_seed(0, "a") == derive_seed(0, "a")
    assert derive_seed(0, "a") != derive_seed(0, "b")
    assert derive_seed(1, "a") != derive_seed(0, "a")
    # Separator prevents ("a","b") colliding with ("ab",).
    assert derive_seed(0, "a", "b") != derive_seed(0, "ab")


def test_same_labels_same_stream() -> None:
    sm = SeedManager(1234)
    a = sm.generator("probe", "noise", 7).normal(size=8)
    b = sm.generator("probe", "noise", 7).normal(size=8)
    assert np.array_equal(a, b)


def test_different_labels_independent() -> None:
    sm = SeedManager(1234)
    a = sm.generator("x").normal(size=8)
    b = sm.generator("y").normal(size=8)
    assert not np.array_equal(a, b)


def test_spawn_is_deterministic() -> None:
    sm = SeedManager(99)
    child1 = sm.spawn("sub").generator("k").integers(0, 1000, size=5)
    child2 = sm.spawn("sub").generator("k").integers(0, 1000, size=5)
    assert np.array_equal(child1, child2)

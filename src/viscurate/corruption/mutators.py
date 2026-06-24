"""Named, replayable function factories for the seven defect injectors (CLAUDE.md §2.2).

The corruption log stores *names* (a mutator name, a domain, a baked value …), never pickled
callables, so a corrupted library is reconstructed deterministically from ``(L0, log)``. This
module is where those names map back to behaviour:

* generic **output mutators** (``roll_h``, ``swap_channels`` …) turn a correct output into a
  plausibly-wrong one for IMPLEMENTATION_BUG and (conditionally) DOMAIN_SCOPED_BUG;
* fn wrappers build the EXACT/PERCEPTUAL duplicates, the fixed-parameter SUBSUMPTION
  specialization, and the cross-family DEAD_SKILL composition.

Everything here is pure numpy — Phase 5 needs no ML dependency.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from viscurate.skills.model import Image, Params, SkillFn

__all__ = [
    "BUG_MUTATORS",
    "DOMAINS",
    "apply_mutator",
    "make_dead_fn",
    "make_domain_bug_fn",
    "make_exact_dup_fn",
    "make_fixed_param_fn",
    "make_impl_bug_fn",
    "make_perceptual_dup_fn",
    "matches_domain",
]

# --------------------------------------------------------------------------------------
# Generic output mutators — each turns a *correct* output into a *wrong-but-valid* one,
# preserving dtype and shape. Named so IMPLEMENTATION_BUG / DOMAIN_SCOPED_BUG are replayable.
# --------------------------------------------------------------------------------------


def _roll_h(out: Image) -> Image:
    """Shift rows by one (off-by-one / wrong-origin in a kernel loop)."""
    return np.roll(out, 1, axis=0)


def _roll_w(out: Image) -> Image:
    """Shift columns by one."""
    return np.roll(out, 1, axis=1)


def _swap_channels(out: Image) -> Image:
    """Swap the first and last channel (a wrong-axis / BGR-vs-RGB style bug)."""
    if out.ndim == 3 and out.shape[2] >= 2:
        swapped = out.copy()
        swapped[..., [0, -1]] = out[..., [-1, 0]]
        return swapped
    return _roll_h(out)  # grayscale fallback so the bug still bites


def _zero_border(out: Image) -> Image:
    """Zero a one-pixel border (a boundary-handling bug)."""
    if out.shape[0] < 3 or out.shape[1] < 3:
        return _roll_h(out)
    bugged = out.copy()
    bugged[0, :] = 0
    bugged[-1, :] = 0
    bugged[:, 0] = 0
    bugged[:, -1] = 0
    return bugged


def _invert_values(out: Image) -> Image:
    """Invert the value range (a sign-swap / missing-negation bug)."""
    if out.dtype == np.bool_:
        return ~out
    if out.dtype == np.uint8:
        return (255 - out.astype(np.int16)).astype(np.uint8)
    if out.dtype == np.uint16:
        return (65535 - out.astype(np.int32)).astype(np.uint16)
    if np.issubdtype(out.dtype, np.floating):
        return np.clip(1.0 - out, 0.0, 1.0).astype(out.dtype)
    return _roll_h(out)


_MUTATORS: dict[str, Callable[[Image], Image]] = {
    "roll_h": _roll_h,
    "roll_w": _roll_w,
    "swap_channels": _swap_channels,
    "zero_border": _zero_border,
    "invert_values": _invert_values,
}

#: Stable, ordered mutator names the planner draws from (order is part of the contract).
BUG_MUTATORS: tuple[str, ...] = (
    "roll_h",
    "roll_w",
    "swap_channels",
    "zero_border",
    "invert_values",
)

#: Domains a DOMAIN_SCOPED_BUG can target — each detectable from the *raw input* array, so the
#: bug fires only when the battery actually contains that domain (CLAUDE.md §2.2 type 6).
DOMAINS: tuple[str, ...] = ("rgba", "grayscale", "uint16")


def _force_one_element(out: Image) -> Image:
    """Last-resort: bump a single element by **one LSB** so the output is not byte-identical.

    Always a *neighbour* step (``max → max-1`` rather than wrapping to ``min``), so the change
    stays ≤1 LSB — keeping the PERCEPTUAL duplicate within tolerance even on a saturated output.
    """
    forced = out.copy()
    if forced.size == 0:
        return forced
    if forced.dtype == np.bool_:
        forced.flat[0] = not bool(forced.flat[0])
    elif np.issubdtype(forced.dtype, np.integer):
        info = np.iinfo(forced.dtype)
        v = int(forced.flat[0])
        forced.flat[0] = v - 1 if v == info.max else v + 1
    elif np.issubdtype(forced.dtype, np.floating):
        fv = float(forced.flat[0])
        forced.flat[0] = fv - 1.0 / 255.0 if fv >= 1.0 else fv + 1.0 / 255.0
    return forced


def _ensure_changed(result: Image, out: Image) -> Image:
    """Guarantee a mutator diverges from its input.

    A fixed mutator can be a no-op for some outputs (``swap_channels`` on a grayscale-replicated
    output; ``roll`` on an axis-constant output). A bug that leaves the output untouched is not a
    bug, so we cascade to a guaranteed-divergent transform — preserving the named flavour in the
    common case while keeping "buggy skills measurably wrong vs the oracle" an invariant.
    """
    if not np.array_equal(result, out):
        return result
    for fallback in (_invert_values, _roll_h, _force_one_element):
        alt = fallback(out)
        if not np.array_equal(alt, out):
            return alt
    return out  # genuinely unrepresentable (e.g. empty output); QA records no divergence


def apply_mutator(name: str, out: Image) -> Image:
    try:
        mutate = _MUTATORS[name]
    except KeyError:
        raise KeyError(f"unknown bug mutator {name!r}; known: {BUG_MUTATORS}") from None
    return _ensure_changed(mutate(out), out)


def matches_domain(image: Image, domain: str) -> bool:
    """Whether the *raw* input array belongs to the targeted domain (pre-coercion)."""
    if domain == "rgba":
        return image.ndim == 3 and image.shape[2] == 4
    if domain == "grayscale":
        return image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)
    if domain == "uint16":
        return image.dtype == np.uint16
    raise KeyError(f"unknown domain {domain!r}; known: {DOMAINS}")


# --------------------------------------------------------------------------------------
# fn-level factories — wrap an L0 callable to realize a defect.
# --------------------------------------------------------------------------------------


def make_impl_bug_fn(base_fn: SkillFn, mutator: str) -> SkillFn:
    """IMPLEMENTATION_BUG: corrupt the output unconditionally (diverges from the oracle)."""

    def buggy(image: Image, params: Params, seed: int) -> Image:
        return apply_mutator(mutator, base_fn(image, params, seed))

    return buggy


def make_domain_bug_fn(base_fn: SkillFn, domain: str, mutator: str) -> SkillFn:
    """DOMAIN_SCOPED_BUG: correct on RGB, corrupt only when the *input* is the bad domain."""

    def domain_buggy(image: Image, params: Params, seed: int) -> Image:
        out = base_fn(image, params, seed)
        return apply_mutator(mutator, out) if matches_domain(image, domain) else out

    return domain_buggy


def make_exact_dup_fn(donor_fn: SkillFn) -> SkillFn:
    """DUPLICATE (exact): the same computation under a new id — byte-identical → EXACT."""

    def dup(image: Image, params: Params, seed: int) -> Image:
        return donor_fn(image, params, seed)

    return dup


def _perceptual_jitter(out: Image) -> Image:
    """A ≤1-LSB deterministic dither: not byte-identical, but PERCEPTUAL-close everywhere.

    A worst-case L∞ ≤ 1/255 (one quantization step), so a PERCEPTUAL duplicate is reliably
    inside any reasonable τ_perceptual yet never byte-identical (→ not EXACT). Where the dither
    clips away (a saturated output), a single 1-LSB neighbour change guarantees non-identity.
    """
    if out.dtype == np.bool_ or out.size == 0:
        return _force_one_element(out)
    h = out.shape[0]
    w = out.shape[1] if out.ndim >= 2 else 1
    rows = np.arange(h).reshape(-1, *([1] * (out.ndim - 1)))
    cols = np.arange(w).reshape(1, -1, *([1] * (out.ndim - 2))) if out.ndim >= 2 else 0
    parity = ((rows + cols) & 1).astype(np.int64)  # broadcasts over channels
    if np.issubdtype(out.dtype, np.integer):
        info = np.iinfo(out.dtype)
        jittered = np.clip(out.astype(np.int64) + parity, info.min, info.max).astype(out.dtype)
    elif np.issubdtype(out.dtype, np.floating):
        step = np.array(1.0 / 255.0, out.dtype)
        jittered = np.clip(out + parity.astype(out.dtype) * step, 0.0, 1.0)
    else:
        return _force_one_element(out)
    return jittered if not np.array_equal(jittered, out) else _force_one_element(out)


def make_perceptual_dup_fn(donor_fn: SkillFn) -> SkillFn:
    """DUPLICATE (perceptual): a numerically-near re-implementation → PERCEPTUAL, not EXACT."""

    def dup(image: Image, params: Params, seed: int) -> Image:
        return _perceptual_jitter(donor_fn(image, params, seed))

    return dup


def make_fixed_param_fn(donor_fn: SkillFn, baked_params: Params) -> SkillFn:
    """SUBSUMPTION: a fixed-parameter specialization (``spec ⊑ donor``).

    ``baked_params`` is the donor's *full* default binding with one value overridden, so the
    specialization (which exposes no params of its own) reproduces exactly one point of the
    donor's grid — the planted ``rotate_90 ⊑ rotate_canvas_degrees`` pattern, generalized.
    """
    frozen = dict(baked_params)

    def specialized(image: Image, params: Params, seed: int) -> Image:
        return donor_fn(image, frozen, seed)

    return specialized


def make_dead_fn(
    inner_fn: SkillFn, inner_params: Params, outer_fn: SkillFn, outer_params: Params
) -> SkillFn:
    """DEAD_SKILL: a correct but useless cross-family composition ``outer(inner(x))``.

    Compositions of two unrelated ops yield a genuinely novel operation that is neither a
    duplicate nor a subsumption of any single library skill — so the verifier sees DISTINCT and
    the skill is removable only on *utility* grounds (CLAUDE.md §3.5.8, type 7).
    """
    inner_frozen, outer_frozen = dict(inner_params), dict(outer_params)

    def dead(image: Image, params: Params, seed: int) -> Image:
        return outer_fn(inner_fn(image, inner_frozen, seed), outer_frozen, seed)

    return dead

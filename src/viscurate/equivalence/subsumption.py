"""Directional subsumption search (CLAUDE.md §3.5.4).

``A ⊑ B`` ("A is subsumed by B") iff for every param binding of A there exists a binding of B
reproducing A's output within EXACT/PERCEPTUAL tolerance **across the whole battery**, but not
conversely. Fixed-param specializations (e.g. ``rotate_90``) expose no params, so A is a single
point and we only search B's grid for a match. The search is a coarse grid with **early-exit
on the first failing probe**, and the direction is recorded.

If both directions hold the pair is EXACT/PERCEPTUAL (handled earlier in the pipeline), so this
function returns :data:`~viscurate.equivalence.relations.Direction.NONE` to avoid double-labels.
Near-miss specializations (a fixed param slightly off) correctly return ``NONE``.
"""

from __future__ import annotations

from dataclasses import dataclass

from viscurate.equivalence.backends import PerceptualBackend, ssim_distance
from viscurate.equivalence.compare import OutputProvider, OutputSet
from viscurate.equivalence.relations import Direction
from viscurate.skills.canonicalize import max_abs_pixel_diff
from viscurate.skills.model import ComparatorView, Params

__all__ = ["SubsumptionResult", "outputs_match", "subsumption_search"]


@dataclass(frozen=True)
class SubsumptionResult:
    direction: Direction
    spec_id: str = ""  # the specialization (the subsumed skill)
    gen_id: str = ""  # the generalization (the subsuming skill)
    binding_map: tuple[tuple[str, str], ...] = ()  # (spec params_key → gen params_key)


def outputs_match(
    spec: OutputSet,
    gen: OutputSet,
    *,
    epsilon: float,
    tau_perceptual: float,
    perceptual: PerceptualBackend | None,
    ssim_floor: float | None,
) -> bool:
    """True iff ``gen`` reproduces ``spec`` within EXACT/PERCEPTUAL tolerance on *every* probe.

    Pixel ``L∞`` is checked first (cheap); LPIPS is computed only for probes that fail the
    pixel test, and the loop **early-exits on the first probe** that fails both — the worst
    case is all that matters for a universally-quantified claim.
    """
    common = spec.common(gen)
    if not common:
        return False
    for p in common:
        if max_abs_pixel_diff(spec.canon[p], gen.canon[p]) <= epsilon:
            continue  # EXACT on this probe
        if perceptual is None:
            return False
        if perceptual.distance(spec.canon[p].rgb, gen.canon[p].rgb) > tau_perceptual:
            return False  # fails both EXACT and PERCEPTUAL → early exit
        if ssim_floor is not None and (
            ssim_distance(spec.canon[p].rgb, gen.canon[p].rgb) > ssim_floor
        ):
            return False
    return True


def _all_reproduced(
    spec_id: str,
    spec_grid: list[Params],
    gen_id: str,
    gen_grid: list[Params],
    provider: OutputProvider,
    *,
    epsilon: float,
    tau_perceptual: float,
    perceptual: PerceptualBackend | None,
    ssim_floor: float | None,
    seed: int | None,
) -> tuple[bool, list[tuple[str, str]]]:
    """Is every ``spec`` binding reproduced by some ``gen`` binding? Returns (ok, binding map)."""
    import json

    mapping: list[tuple[str, str]] = []
    for sp in spec_grid:
        spec_out = provider.outputs(spec_id, sp, seed=seed)
        found = False
        for gp in gen_grid:
            gen_out = provider.outputs(gen_id, gp, seed=seed)
            if outputs_match(
                spec_out,
                gen_out,
                epsilon=epsilon,
                tau_perceptual=tau_perceptual,
                perceptual=perceptual,
                ssim_floor=ssim_floor,
            ):
                mapping.append((json.dumps(sp, sort_keys=True), json.dumps(gp, sort_keys=True)))
                found = True
                break
        if not found:
            return False, []
    return True, mapping


def subsumption_search(
    view_a: ComparatorView,
    view_b: ComparatorView,
    provider: OutputProvider,
    *,
    grid_a: list[Params],
    grid_b: list[Params],
    epsilon: float,
    tau_perceptual: float,
    perceptual: PerceptualBackend | None = None,
    ssim_floor: float | None = None,
    seed: int | None = None,
) -> SubsumptionResult:
    """Test both directions; certify the one that holds exclusively (else ``NONE``)."""
    a_sub_b, map_ab = _all_reproduced(
        view_a.id,
        grid_a,
        view_b.id,
        grid_b,
        provider,
        epsilon=epsilon,
        tau_perceptual=tau_perceptual,
        perceptual=perceptual,
        ssim_floor=ssim_floor,
        seed=seed,
    )
    b_sub_a, map_ba = _all_reproduced(
        view_b.id,
        grid_b,
        view_a.id,
        grid_a,
        provider,
        epsilon=epsilon,
        tau_perceptual=tau_perceptual,
        perceptual=perceptual,
        ssim_floor=ssim_floor,
        seed=seed,
    )
    if a_sub_b and not b_sub_a:  # A ⊑ B : B is the generalization
        return SubsumptionResult(Direction.B_SUBSUMES_A, view_a.id, view_b.id, tuple(map_ab))
    if b_sub_a and not a_sub_b:  # B ⊑ A : A is the generalization
        return SubsumptionResult(Direction.A_SUBSUMES_B, view_b.id, view_a.id, tuple(map_ba))
    return SubsumptionResult(Direction.NONE)

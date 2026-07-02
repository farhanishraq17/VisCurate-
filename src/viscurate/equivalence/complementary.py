"""The COMPLEMENTARY detector (CLAUDE.md §3.5.4).

The docs leave COMPLEMENTARY abstract; this makes it operational. A pair is complementary if it
operates on **disjoint image aspects** and composes meaningfully. Three conditions must hold:

1. **Both non-trivial** — each differs from identity on the battery (rules out a no-op).
2. **Not equivalent / subsuming** — guaranteed by the caller (stop-at-first reached this stage).
3. **Approximate commutation** — ``D(A(B(x)), B(A(x)))`` is EXACT/PERCEPTUAL-small across the
   battery: order does not matter, the hallmark of orthogonal ops (geometry × colour).

This requires the verifier to **execute compositions** (a small added capability). Where the
composition is shape-incompatible or order clearly matters, the test fails and the pair is
DISTINCT. The relation is decided from *outputs only* — never from metadata/family tags, which
would smuggle text back into the output-grounded path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from viscurate.equivalence.backends import PerceptualBackend
from viscurate.equivalence.compare import OutputProvider, OutputSet, worst_case
from viscurate.equivalence.subsumption import outputs_match
from viscurate.skills.canonicalize import max_abs_pixel_diff
from viscurate.skills.model import ComparatorView, Params

__all__ = ["ComplementaryResult", "is_complementary", "is_trivial"]


@dataclass(frozen=True)
class ComplementaryResult:
    is_complementary: bool
    reason: str
    commute_linf: float = float("nan")
    commute_lpips: float = float("nan")
    effect_corr: float = float("nan")
    worst_probe: str = ""


def is_trivial(
    skill_id: str,
    params: Params | None,
    provider: OutputProvider,
    *,
    epsilon: float,
    seed: int | None = None,
) -> bool:
    """True iff the skill ≈ identity on *every* probe (a no-op): max pixel ``L∞`` ≤ ε."""
    out = provider.outputs(skill_id, params, seed=seed)
    ident = provider.identity_outputs()
    common = out.common(ident)
    if not common:
        return False
    return all(max_abs_pixel_diff(out.canon[p], ident.canon[p]) <= epsilon for p in common)


def is_complementary(
    view_a: ComparatorView,
    view_b: ComparatorView,
    provider: OutputProvider,
    *,
    params_a: Params | None = None,
    params_b: Params | None = None,
    epsilon: float,
    commute_tau: float,
    trivial_epsilon: float,
    perceptual: PerceptualBackend | None = None,
    seed: int | None = None,
) -> ComplementaryResult:
    """Decide COMPLEMENTARY by non-triviality + approximate commutation (defaults if unset)."""
    pa = provider.default_params(view_a.id) if params_a is None else params_a
    pb = provider.default_params(view_b.id) if params_b is None else params_b

    if is_trivial(view_a.id, pa, provider, epsilon=trivial_epsilon, seed=seed):
        return ComplementaryResult(False, f"{view_a.id} is a no-op at the tested binding")
    if is_trivial(view_b.id, pb, provider, epsilon=trivial_epsilon, seed=seed):
        return ComplementaryResult(False, f"{view_b.id} is a no-op at the tested binding")

    ab: OutputSet = provider.compose_outputs(view_a.id, pa, view_b.id, pb, seed=seed)
    ba: OutputSet = provider.compose_outputs(view_b.id, pb, view_a.id, pa, seed=seed)
    if not ab.common(ba):
        return ComplementaryResult(False, "compositions are shape-incompatible → DISTINCT")

    linf = worst_case({p: max_abs_pixel_diff(ab.canon[p], ba.canon[p]) for p in ab.common(ba)})
    lpips_val = float("nan")
    if perceptual is not None and linf.value > epsilon:
        lpips_val = worst_case(
            {p: perceptual.distance(ab.canon[p].rgb, ba.canon[p].rgb) for p in ab.common(ba)}
        ).value

    commutes = outputs_match(
        ab, ba, epsilon=epsilon, tau_perceptual=commute_tau, perceptual=perceptual, ssim_floor=None
    )
    corr = _effect_residual_correlation(view_a.id, pa, view_b.id, pb, provider, seed=seed)
    same_aspect = abs(corr) > 0.85 if not np.isnan(corr) else False
    reason = (
        f"compositions commute (L∞ {linf.value:.4f}, residual corr {corr:.3f})"
        if commutes
        else f"order matters (L∞ {linf.value:.4f} @ {linf.probe_id}) → DISTINCT"
    )
    if commutes and same_aspect:
        return ComplementaryResult(
            False,
            f"compositions commute but residual effects are same-aspect (corr {corr:.3f})",
            linf.value,
            lpips_val,
            corr,
            linf.probe_id,
        )
    return ComplementaryResult(commutes, reason, linf.value, lpips_val, corr, linf.probe_id)


def _effect_residual_correlation(
    a_id: str,
    params_a: Params,
    b_id: str,
    params_b: Params,
    provider: OutputProvider,
    *,
    seed: int | None,
) -> float:
    ident = provider.identity_outputs()
    ao = provider.outputs(a_id, params_a, seed=seed)
    bo = provider.outputs(b_id, params_b, seed=seed)
    vals_a: list[np.ndarray] = []
    vals_b: list[np.ndarray] = []
    for probe_id in sorted(set(ident.probe_ids) & set(ao.probe_ids) & set(bo.probe_ids)):
        ia = ident.canon[probe_id].rgb.astype(np.float32)
        aa = ao.canon[probe_id].rgb.astype(np.float32)
        bb = bo.canon[probe_id].rgb.astype(np.float32)
        if ia.shape != aa.shape or ia.shape != bb.shape:
            continue
        vals_a.append((aa - ia).reshape(-1))
        vals_b.append((bb - ia).reshape(-1))
    if not vals_a:
        return float("nan")
    ra = np.concatenate(vals_a)
    rb = np.concatenate(vals_b)
    if float(np.linalg.norm(ra)) == 0.0 or float(np.linalg.norm(rb)) == 0.0:
        return float("nan")
    return float(np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))

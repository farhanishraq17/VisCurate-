"""The hierarchical stop-at-first taxonomy engine (CLAUDE.md §3.5.3).

``classify(A, B)`` returns exactly one :class:`RelationResult` by walking the relations in
cheapest-and-strictest order — EXACT → PERCEPTUAL → SUBSUMPTION → SEMANTIC_PRESERVING →
COMPLEMENTARY → DISTINCT — and stopping at the first that holds. A calibrated **abstention
band** ``[τ(1−δ), τ(1+δ)]`` around the PERCEPTUAL/SEMANTIC thresholds returns **UNCERTAIN**
(routing the pair to human review) instead of forcing a class.

Design decisions made concrete here (CLAUDE.md §3.5.1, §3.5.3):

* EXACT/PERCEPTUAL aggregate **worst-case (max)** over the matched sweep × probes — equivalence
  is universally quantified. SEMANTIC aggregates a **high quantile (p90)** plus the mean.
* Order: SUBSUMPTION is checked *before* SEMANTIC, since a true directional subsumption is a
  stronger, actionable relation that a loose semantic match must not mask.
* The SSIM floor guards LPIPS blind spots: a pair below the LPIPS band but failing the SSIM
  floor is *not* PERCEPTUAL and falls through.

Backends are optional: with ``perceptual=None`` the PERCEPTUAL stage is skipped and subsumption
runs EXACT-only; with ``semantic=None`` the SEMANTIC stage is skipped. This keeps the engine
exercisable without the ``[ml]`` extra (e.g. EXACT/SUBSUMPTION/COMPLEMENTARY/DISTINCT tests).
"""

from __future__ import annotations

from viscurate.config import ThresholdConfig
from viscurate.equivalence.backends import (
    PerceptualBackend,
    SemanticBackend,
    cosine_distance,
    ssim_distance,
)
from viscurate.equivalence.compare import (
    Aggregate,
    OutputProvider,
    mean,
    quantile,
)
from viscurate.equivalence.complementary import is_complementary
from viscurate.equivalence.param_alignment import ParamAlignment
from viscurate.equivalence.relations import Direction, Relation, RelationResult
from viscurate.equivalence.subsumption import subsumption_search
from viscurate.skills.canonicalize import content_hash, max_abs_pixel_diff
from viscurate.skills.model import ComparatorView, Params

__all__ = ["classify"]


def _sweep(
    alignment: ParamAlignment | None, provider: OutputProvider, a_id: str, b_id: str
) -> list[tuple[Params, Params]]:
    if alignment is not None:
        ms = alignment.matched_sweep(a_id, b_id)
        if ms:
            return ms
    return [(provider.default_params(a_id), provider.default_params(b_id))]


def _subsumption_grid(
    alignment: ParamAlignment | None, provider: OutputProvider, skill_id: str
) -> list[Params]:
    if alignment is not None:
        g = alignment.subsumption_grid(skill_id)
        if g is not None:
            return g
    return provider.param_grid(skill_id, max_points=24)


def _sweep_pixel_worst(
    a_id: str,
    b_id: str,
    sweep: list[tuple[Params, Params]],
    provider: OutputProvider,
    seed: int | None,
) -> tuple[Aggregate, bool]:
    """Worst pixel ``L∞`` over the sweep × probes, plus whether every point is hash-identical."""
    agg = Aggregate(value=-1.0, probe_id="")
    all_hash = True
    saw_common = False
    for pa, pb in sweep:
        ao = provider.outputs(a_id, pa, seed=seed)
        bo = provider.outputs(b_id, pb, seed=seed)
        common = ao.common(bo)
        if common:
            saw_common = True
        for p in common:
            if content_hash(ao.canon[p]) != content_hash(bo.canon[p]):
                all_hash = False
            d = max_abs_pixel_diff(ao.canon[p], bo.canon[p])
            if d > agg.value:
                agg = Aggregate(value=d, probe_id=p)
    if not saw_common:
        return Aggregate(value=float("inf"), probe_id=""), False
    return agg, all_hash


def _sweep_perceptual_worst(
    a_id: str,
    b_id: str,
    sweep: list[tuple[Params, Params]],
    provider: OutputProvider,
    perceptual: PerceptualBackend,
    seed: int | None,
) -> tuple[Aggregate, Aggregate]:
    """Worst LPIPS and worst ``1−SSIM`` over the sweep × probes."""
    lp = Aggregate(value=-1.0, probe_id="")
    ss = Aggregate(value=-1.0, probe_id="")
    for pa, pb in sweep:
        ao = provider.outputs(a_id, pa, seed=seed)
        bo = provider.outputs(b_id, pb, seed=seed)
        for p in ao.common(bo):
            dl = perceptual.distance(ao.canon[p].rgb, bo.canon[p].rgb)
            if dl > lp.value:
                lp = Aggregate(value=dl, probe_id=p)
            ds = ssim_distance(ao.canon[p].rgb, bo.canon[p].rgb)
            if ds > ss.value:
                ss = Aggregate(value=ds, probe_id=p)
    return lp, ss


def _sweep_semantic(
    a_id: str,
    b_id: str,
    sweep: list[tuple[Params, Params]],
    provider: OutputProvider,
    semantic: SemanticBackend,
    clip: SemanticBackend | None,
    seed: int | None,
) -> dict[str, float]:
    """Per (point, probe) DINO cosine distance; CLIP taken as the larger, more conservative view."""
    out: dict[str, float] = {}
    for i, (pa, pb) in enumerate(sweep):
        ao = provider.outputs(a_id, pa, seed=seed)
        bo = provider.outputs(b_id, pb, seed=seed)
        common = ao.common(bo)
        if not common:
            continue
        fa = semantic.features([ao.canon[p].rgb for p in common])
        fb = semantic.features([bo.canon[p].rgb for p in common])
        ca = clip.features([ao.canon[p].rgb for p in common]) if clip is not None else None
        cb = clip.features([bo.canon[p].rgb for p in common]) if clip is not None else None
        for j, p in enumerate(common):
            d = cosine_distance(fa[j], fb[j])
            if ca is not None and cb is not None:
                d = max(d, cosine_distance(ca[j], cb[j]))
            out[f"{i}:{p}"] = d
    return out


def classify(
    view_a: ComparatorView,
    view_b: ComparatorView,
    provider: OutputProvider,
    *,
    thresholds: ThresholdConfig,
    perceptual: PerceptualBackend | None = None,
    semantic: SemanticBackend | None = None,
    clip: SemanticBackend | None = None,
    alignment: ParamAlignment | None = None,
    seed: int | None = None,
) -> RelationResult:
    """Classify the pair ``(A, B)`` into one relation (or UNCERTAIN), from outputs only."""
    a_id, b_id = view_a.id, view_b.id
    eps = thresholds.exact_epsilon
    tau_p = thresholds.perceptual_lpips
    tau_s = thresholds.semantic_dino
    delta = thresholds.abstention_delta
    sweep = _sweep(alignment, provider, a_id, b_id)

    # -- Stage 1: EXACT -------------------------------------------------------------
    pix, all_hash = _sweep_pixel_worst(a_id, b_id, sweep, provider, seed)
    if all_hash and pix.value != float("inf"):
        return RelationResult(
            Relation.EXACT,
            reason="hash-identical on every matched probe",
            distances={"l_inf": 0.0},
            alternatives=("merge",),
        )
    if pix.value <= eps:
        return RelationResult(
            Relation.EXACT,
            reason=f"worst-case L∞ {pix.value:.5f} ≤ ε ({eps:.5f})",
            distances={"l_inf": pix.value},
            worst_probe=pix.probe_id,
            alternatives=("merge",),
        )

    # -- Stage 2: PERCEPTUAL --------------------------------------------------------
    if perceptual is not None:
        lp, ss = _sweep_perceptual_worst(a_id, b_id, sweep, provider, perceptual, seed)
        dist = {"l_inf": pix.value, "lpips": lp.value, "ssim_dist": ss.value}
        if lp.value <= tau_p * (1.0 - delta):
            if ss.value <= thresholds.perceptual_ssim:
                return RelationResult(
                    Relation.PERCEPTUAL,
                    reason=f"worst-case LPIPS {lp.value:.4f} ≤ τ_perc, SSIM floor ok",
                    distances=dist,
                    worst_probe=lp.probe_id,
                    alternatives=("merge", "parameterize"),
                )
            # below the LPIPS band but fails the SSIM floor → LPIPS blind spot; fall through.
        elif lp.value <= tau_p * (1.0 + delta):
            return RelationResult(
                Relation.UNCERTAIN,
                reason=f"LPIPS {lp.value:.4f} in abstention band around τ_perc={tau_p}",
                distances=dist,
                worst_probe=lp.probe_id,
                uncertain_about=Relation.PERCEPTUAL,
            )

    # -- Stage 3: SUBSUMPTION (directional) -----------------------------------------
    grid_a = _subsumption_grid(alignment, provider, a_id)
    grid_b = _subsumption_grid(alignment, provider, b_id)
    sub = subsumption_search(
        view_a,
        view_b,
        provider,
        grid_a=grid_a,
        grid_b=grid_b,
        epsilon=eps,
        tau_perceptual=tau_p,
        perceptual=perceptual,
        ssim_floor=thresholds.perceptual_ssim if perceptual is not None else None,
        seed=seed,
    )
    if sub.direction is not Direction.NONE:
        rel = "⊑" if sub.direction is Direction.B_SUBSUMES_A else "⊒"
        return RelationResult(
            Relation.SUBSUMPTION,
            direction=sub.direction,
            reason=f"{sub.spec_id} {rel} {sub.gen_id} (every specialization binding reproduced)",
            distances={"l_inf": pix.value},
            alternatives=("parameterize", "keep_separate"),
        )

    # -- Stage 4: SEMANTIC_PRESERVING -----------------------------------------------
    if semantic is not None:
        dino = _sweep_semantic(a_id, b_id, sweep, provider, semantic, clip, seed)
        p90 = quantile(dino, thresholds.semantic_quantile)
        mn = mean(dino)
        dist = {"dino_p90": p90, "dino_mean": mn}
        if p90 <= tau_s * (1.0 - delta) and mn <= tau_s:
            return RelationResult(
                Relation.SEMANTIC_PRESERVING,
                reason=f"DINO p90 {p90:.4f} ≤ τ_sem, mean {mn:.4f} (human-verified slice)",
                distances=dist,
                alternatives=("parameterize", "keep_separate"),
            )
        if p90 <= tau_s * (1.0 + delta):
            return RelationResult(
                Relation.UNCERTAIN,
                reason=f"DINO p90 {p90:.4f} in abstention band around τ_sem={tau_s}",
                distances=dist,
                uncertain_about=Relation.SEMANTIC_PRESERVING,
            )

    # -- Stage 5: COMPLEMENTARY -----------------------------------------------------
    comp = is_complementary(
        view_a,
        view_b,
        provider,
        epsilon=eps,
        commute_tau=thresholds.complementary_lpips,
        trivial_epsilon=eps,
        perceptual=perceptual,
        seed=seed,
    )
    if comp.is_complementary:
        return RelationResult(
            Relation.COMPLEMENTARY,
            reason=comp.reason,
            distances={"commute_linf": comp.commute_linf, "commute_lpips": comp.commute_lpips},
            worst_probe=comp.worst_probe,
            alternatives=("keep_separate",),
        )

    # -- Stage 6: DISTINCT (residual) -----------------------------------------------
    return RelationResult(
        Relation.DISTINCT,
        reason=f"failed every relation; worst-case L∞ {pix.value:.4f} @ {pix.probe_id}",
        distances={"l_inf": pix.value},
        worst_probe=pix.probe_id,
        alternatives=("keep_separate", "parameterize"),
    )

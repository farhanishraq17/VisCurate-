"""Phase 3 — the output-grounded equivalence engine.

The taxonomy is driven by *deterministic fake backends* (no ML dependency) so each relation
branch is exercised exactly and reproducibly; the real LPIPS/DINO/CLIP smoke test lives in
``test_equivalence_ml.py`` (gated on the ``[ml]`` extra). The hand-built EXACT / PERCEPTUAL /
DISTINCT pairs here are the Phase-3 exit criterion.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from viscurate.config import ThresholdConfig
from viscurate.equivalence import (
    BatteryEvaluator,
    Direction,
    LabeledPair,
    ParamAlignment,
    Relation,
    calibrate_thresholds,
    candidate_pairs,
    classify,
    compute_fingerprints,
    select_threshold,
    subsumption_search,
)
from viscurate.equivalence.param_alignment import load_param_alignment
from viscurate.skills.library._build import make_skill
from viscurate.skills.model import Image, Params, Skill

# --------------------------------------------------------------------------------------------
# Deterministic fake backends — no torch. The fakes are intentionally simple proxies whose
# values we can reason about exactly: LPIPS ≈ mean |Δ|, DINO feature ≈ mean RGB.
# --------------------------------------------------------------------------------------------


class FakePerceptual:
    name = "fake-perc"

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return float("inf")
        return float(np.mean(np.abs(a - b)))


class FakeSemantic:
    name = "fake-sem"

    def features(self, imgs: Sequence[np.ndarray]) -> np.ndarray:
        if not imgs:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in imgs])


# --------------------------------------------------------------------------------------------
# Test skills (tiny, fully controlled) + batteries.
# --------------------------------------------------------------------------------------------


def _ident(image: Image, params: Params, seed: int) -> Image:
    return image.copy()


def _add(delta: int):  # type: ignore[no-untyped-def]
    def fn(image: Image, params: Params, seed: int) -> Image:
        return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    return fn


def _mul(factor: float):  # type: ignore[no-untyped-def]
    def fn(image: Image, params: Params, seed: int) -> Image:
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return fn


def _flip(image: Image, params: Params, seed: int) -> Image:
    return np.ascontiguousarray(image[:, ::-1])


def _invert(image: Image, params: Params, seed: int) -> Image:
    return (255 - image).astype(np.uint8)


def _skill(skill_id: str, fn, family: str = "test") -> Skill:  # type: ignore[no-untyped-def]
    return make_skill(skill_id, skill_id, "secret description", fn, family)


def _textured_battery() -> list[tuple[str, Image]]:
    rng = np.random.default_rng(7)
    tex = rng.integers(20, 200, size=(48, 48, 3), dtype=np.uint8)  # mid-range → no clipping
    grad = np.tile(np.linspace(20, 200, 48, dtype=np.uint8)[None, :, None], (48, 1, 3))
    return [("tex", tex), ("grad", grad)]


def _uniform_battery(value: int = 100) -> list[tuple[str, Image]]:
    return [("flat", np.full((32, 32, 3), value, dtype=np.uint8))]


def _evaluator(skills: list[Skill], battery: list[tuple[str, Image]]) -> BatteryEvaluator:
    return BatteryEvaluator(skills, battery)


def test_bounded_output_cache_evicts_but_preserves_results() -> None:
    # The output cache is a compute optimization: bounding it (LRU) must cap memory without
    # changing any result — a cache miss just recomputes the (deterministic) output.
    skills = [
        _skill("s_flip", _flip),
        _skill("s_inv", _invert),
        _skill("s_add", _add(3)),
        _skill("s_mul", _mul(2.0)),
        _skill("s_id", _ident),
    ]
    battery = _textured_battery()
    ref = BatteryEvaluator(skills, battery)  # unbounded reference
    bounded = BatteryEvaluator(skills, battery, max_cache_entries=2)

    for s in skills:
        want = ref.outputs(s.id)
        got = bounded.outputs(s.id)
        assert got.probe_ids == want.probe_ids
        for p in got.probe_ids:
            assert np.array_equal(got.raw[p], want.raw[p])  # identical despite eviction
        assert len(bounded._cache) <= 2  # never exceeds the LRU bound

    # 5 distinct skills through a size-2 cache ⇒ eviction actually happened…
    assert len(bounded._cache) == 2
    # …and re-querying an evicted skill recomputes the same result (cache-miss path).
    again = bounded.outputs(skills[0].id)
    want0 = ref.outputs(skills[0].id)
    for p in again.probe_ids:
        assert np.array_equal(again.raw[p], want0.raw[p])


# --------------------------------------------------------------------------------------------
# EXACT / PERCEPTUAL / DISTINCT — the exit-criterion hand-built pairs.
# --------------------------------------------------------------------------------------------


def test_exact_duplicate_different_id() -> None:
    flip = _skill("flip_v1", _flip, "geo")
    clone = _skill("flip_clone_v1", _flip, "geo")  # same fn, different id → EXACT duplicate
    ev = _evaluator([flip, clone], _textured_battery())
    res = classify(
        flip.comparator_view(), clone.comparator_view(), ev, thresholds=ThresholdConfig()
    )
    assert res.relation is Relation.EXACT
    assert res.licenses_merge


def test_perceptual_small_numeric_difference() -> None:
    ident = _skill("ident_v1", _ident)
    add3 = _skill("add3_v1", _add(3))  # +3/255 ≈ 0.0118 L∞ (> ε), tiny LPIPS, SSIM ~ 1
    ev = _evaluator([ident, add3], _textured_battery())
    res = classify(
        ident.comparator_view(),
        add3.comparator_view(),
        ev,
        thresholds=ThresholdConfig(),
        perceptual=FakePerceptual(),
    )
    assert res.relation is Relation.PERCEPTUAL
    assert res.distances["lpips"] < ThresholdConfig().perceptual_lpips


def test_distinct_non_commuting_pair() -> None:
    addbig = _skill("addbig_v1", _add(80))
    mul3 = _skill("mul3_v1", _mul(3.0))  # clipping makes the order matter → no commute
    ev = _evaluator([addbig, mul3], _textured_battery())
    res = classify(
        addbig.comparator_view(),
        mul3.comparator_view(),
        ev,
        thresholds=ThresholdConfig(),
        perceptual=FakePerceptual(),
        semantic=None,
    )
    assert res.relation is Relation.DISTINCT
    assert res.worst_probe in ("tex", "grad")


def test_matched_sweep_worst_case_blocks_false_merge() -> None:
    # blur_gaussian vs blur_box agree at small k, diverge at large k → the matched-sweep
    # worst-case must NOT license a merge (EXACT/PERCEPTUAL) — the silent-merge bug the project
    # exists to prevent. (Whether the residual label is SEMANTIC/COMPLEMENTARY/DISTINCT is a
    # backend-calibration question for Phase 4; the safety property is "no false merge".)
    from viscurate.skills.library import build_builtin_registry

    reg = build_builtin_registry()
    g, b = reg.get("blur_gaussian_v1"), reg.get("blur_box_v1")
    align = load_param_alignment("configs/param_alignment.yaml")
    ev = _evaluator([g, b], _textured_battery())
    res = classify(
        g.comparator_view(),
        b.comparator_view(),
        ev,
        thresholds=ThresholdConfig(),
        perceptual=FakePerceptual(),
        semantic=FakeSemantic(),
        alignment=align,
    )
    assert res.relation not in (Relation.EXACT, Relation.PERCEPTUAL)
    assert not res.licenses_merge


# --------------------------------------------------------------------------------------------
# SUBSUMPTION (directional) and the near-miss negative.
# --------------------------------------------------------------------------------------------


def test_subsumption_rotate90_subsumed_by_rotate_canvas() -> None:
    from viscurate.skills.library import build_builtin_registry

    reg = build_builtin_registry()
    r90, rc = reg.get("rotate_90_v1"), reg.get("rotate_canvas_degrees_v1")
    align = load_param_alignment("configs/param_alignment.yaml")
    ev = _evaluator([r90, rc], _textured_battery())
    res = classify(
        r90.comparator_view(),
        rc.comparator_view(),
        ev,
        thresholds=ThresholdConfig(),
        perceptual=FakePerceptual(),
        alignment=align,
    )
    assert res.relation is Relation.SUBSUMPTION
    assert res.direction is Direction.B_SUBSUMES_A  # rotate_90 ⊑ rotate_canvas_degrees


def test_subsumption_search_near_miss_returns_none() -> None:
    from viscurate.skills.library import build_builtin_registry

    reg = build_builtin_registry()
    r90, r180 = reg.get("rotate_90_v1"), reg.get("rotate_180_v1")
    ev = _evaluator([r90, r180], _textured_battery())
    out = subsumption_search(
        r90.comparator_view(),
        r180.comparator_view(),
        ev,
        grid_a=[{}],
        grid_b=[{}],
        epsilon=ThresholdConfig().exact_epsilon,
        tau_perceptual=ThresholdConfig().perceptual_lpips,
    )
    assert out.direction is Direction.NONE


def test_subsumption_crop_center_subsumed_by_bounding_box() -> None:
    from viscurate.skills.library import build_builtin_registry

    reg = build_builtin_registry()
    cc, bb = reg.get("crop_center_percentage_v1"), reg.get("crop_bounding_box_v1")
    align = load_param_alignment("configs/param_alignment.yaml")
    # Even-sided probe so center-crop rounding aligns exactly with the fractional bbox.
    ev = _evaluator(
        [cc, bb], [("even", np.random.default_rng(3).integers(0, 256, (64, 64, 3), dtype=np.uint8))]
    )
    res = classify(
        cc.comparator_view(),
        bb.comparator_view(),
        ev,
        thresholds=ThresholdConfig(),
        perceptual=FakePerceptual(),
        alignment=align,
    )
    assert res.relation is Relation.SUBSUMPTION
    assert res.direction is Direction.B_SUBSUMES_A  # crop_center ⊑ crop_bounding_box


# --------------------------------------------------------------------------------------------
# SEMANTIC_PRESERVING, COMPLEMENTARY, and the UNCERTAIN abstention band.
# --------------------------------------------------------------------------------------------


def test_semantic_preserving_when_perceptually_far_but_feature_close() -> None:
    add3 = _skill("add3_v1", _add(3))
    add5 = _skill("add5_v1", _add(5))
    ev = _evaluator([add3, add5], _textured_battery())
    # τ_perc tiny → fails PERCEPTUAL; τ_sem generous → mean-RGB features are ~parallel.
    th = ThresholdConfig(perceptual_lpips=0.001, semantic_dino=0.5)
    res = classify(
        add3.comparator_view(),
        add5.comparator_view(),
        ev,
        thresholds=th,
        perceptual=FakePerceptual(),
        semantic=FakeSemantic(),
    )
    assert res.relation is Relation.SEMANTIC_PRESERVING


def test_complementary_flip_and_invert_commute() -> None:
    flip = _skill("flip_v1", _flip, "geo")
    invert = _skill("invert_v1", _invert, "color")
    ev = _evaluator([flip, invert], _textured_battery())
    res = classify(
        flip.comparator_view(),
        invert.comparator_view(),
        ev,
        thresholds=ThresholdConfig(),
        perceptual=FakePerceptual(),
        semantic=None,
    )
    assert res.relation is Relation.COMPLEMENTARY
    assert abs(res.distances["effect_corr"]) < 0.85


def test_commuting_same_aspect_effects_are_not_complementary() -> None:
    add10 = _skill("add10_v1", _add(10))
    add20 = _skill("add20_v1", _add(20))
    ev = _evaluator([add10, add20], _textured_battery())
    th = ThresholdConfig(perceptual_lpips=0.001, semantic_dino=0.001)
    res = classify(
        add10.comparator_view(),
        add20.comparator_view(),
        ev,
        thresholds=th,
        perceptual=FakePerceptual(),
        semantic=None,
    )
    assert res.relation is Relation.DISTINCT


def test_abstention_band_returns_uncertain() -> None:
    add3 = _skill("add3_v1", _add(3))
    add5 = _skill("add5_v1", _add(5))
    ev = _evaluator([add3, add5], _uniform_battery(100))  # exact ΔLPIPS = 2/255
    th = ThresholdConfig(perceptual_lpips=2.0 / 255.0, abstention_delta=0.10)
    res = classify(
        add3.comparator_view(),
        add5.comparator_view(),
        ev,
        thresholds=th,
        perceptual=FakePerceptual(),
    )
    assert res.relation is Relation.UNCERTAIN
    assert res.uncertain_about is Relation.PERCEPTUAL


# --------------------------------------------------------------------------------------------
# The modality boundary (CLAUDE.md §1.2) — the comparator view carries no description.
# --------------------------------------------------------------------------------------------


def test_comparator_view_has_no_description() -> None:
    s = _skill("secret_v1", _ident)
    assert s.description == "secret description"
    view = s.comparator_view()
    assert not hasattr(view, "description")


# --------------------------------------------------------------------------------------------
# Output-based candidate generation.
# --------------------------------------------------------------------------------------------


def test_candidate_generation_collides_same_output_and_keeps_boundaries() -> None:
    # Two skills with identical outputs but unrelated ids/families must collide on fingerprints.
    flip = _skill("flip_v1", _flip, "geo")
    flip_twin = _skill("mystery_v1", _flip, "stylize")
    addbig = _skill("addbig_v1", _add(80), "color")
    ev = _evaluator([flip, flip_twin, addbig], _textured_battery())
    views = [s.comparator_view() for s in (flip, flip_twin, addbig)]
    fps = compute_fingerprints(views, ev, FakeSemantic(), screening_ids=["tex", "grad"])
    pairs = candidate_pairs(views, fps, hard_negatives=[("flip_v1", "addbig_v1")])
    assert ("flip_v1", "mystery_v1") in pairs  # identical output → output-based collision
    assert ("addbig_v1", "flip_v1") in pairs  # engineered hard negative always kept


def test_same_family_pairs_always_included() -> None:
    a = _skill("a_v1", _add(3), "fam")
    b = _skill("b_v1", _add(80), "fam")
    ev = _evaluator([a, b], _textured_battery())
    views = [s.comparator_view() for s in (a, b)]
    fps = compute_fingerprints(views, ev, FakeSemantic(), screening_ids=["tex", "grad"])
    pairs = candidate_pairs(views, fps, k=0, max_distance=-1.0, hard_negatives=[])
    assert ("a_v1", "b_v1") in pairs  # same family → always a candidate


# --------------------------------------------------------------------------------------------
# Threshold calibration procedure.
# --------------------------------------------------------------------------------------------


def test_select_threshold_maximizes_recall_under_precision_floor() -> None:
    # positives (mergeable) have low scores; one negative sits at 0.06.
    samples = [(0.01, True), (0.02, True), (0.04, True), (0.06, False), (0.20, False)]
    fit = select_threshold(samples, min_precision=1.0, min_recall=0.5)
    assert fit.met_target
    assert 0.04 <= fit.threshold < 0.06  # admits all positives, excludes the 0.06 negative
    assert fit.precision == 1.0
    assert fit.recall == 1.0


def test_calibrate_thresholds_stamps_provenance() -> None:
    pairs = [
        LabeledPair(Relation.EXACT, {"lpips": 0.01, "dino_p90": 0.02}),
        LabeledPair(Relation.PERCEPTUAL, {"lpips": 0.03, "dino_p90": 0.05}),
        LabeledPair(Relation.DISTINCT, {"lpips": 0.40, "dino_p90": 0.60}),
        LabeledPair(Relation.SEMANTIC_PRESERVING, {"lpips": 0.30, "dino_p90": 0.10}),
    ]
    calibrated, fits = calibrate_thresholds(
        pairs, base=ThresholdConfig(), date="2026-06-24", min_recall=0.5
    )
    assert calibrated.calibrated is True
    assert calibrated.calibration_date == "2026-06-24"
    assert calibrated.calibration_split_hash and len(calibrated.calibration_split_hash) == 64
    assert "perceptual_lpips" in fits
    # The calibrated config must itself satisfy the provenance rule (constructs without error).
    assert calibrated.perceptual_lpips > 0


# --------------------------------------------------------------------------------------------
# param_alignment artifact.
# --------------------------------------------------------------------------------------------


def test_param_alignment_loads_and_matches_blur_axis() -> None:
    align = load_param_alignment("configs/param_alignment.yaml")
    assert isinstance(align, ParamAlignment)
    sweep = align.matched_sweep("blur_gaussian_v1", "blur_box_v1")
    assert sweep is not None and len(sweep) == 7  # the kernel_size grid
    # rotate_canvas subsumption grid must include the 90° quarter-turn.
    grid = align.subsumption_grid("rotate_canvas_degrees_v1")
    assert grid is not None
    assert any(p.get("degrees") == 90.0 for p in grid)


def test_aggregation_worst_case_and_quantile() -> None:
    from viscurate.equivalence.compare import mean, quantile, worst_case

    d = {"a": 0.1, "b": 0.9, "c": 0.2}
    agg = worst_case(d)
    assert agg.value == pytest.approx(0.9) and agg.probe_id == "b"
    assert quantile(d, 0.5) == pytest.approx(0.2)
    assert mean(d) == pytest.approx(0.4)
    assert worst_case({}).value == float("inf")

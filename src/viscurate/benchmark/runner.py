"""The equivalence-benchmark runner (CLAUDE.md Phase 4, §3.5.9).

End to end: build the candidate pairs (output-based generation ∪ the planted ``G0`` structure),
run the **output-grounded verifier** and the **text baselines** over them, measure the full
distance vector per pair (for threshold calibration), and assemble a :class:`BenchmarkResult`
the report/metrics consume.

The two judge families stay strictly separated:

* the output track goes through :func:`viscurate.equivalence.classify`, which is handed a
  :class:`~viscurate.skills.model.ComparatorView` + an
  :class:`~viscurate.equivalence.compare.OutputProvider` — it never sees text;
* the text tracks come from :mod:`viscurate.baselines`, which reads ``SkillSpec`` text.

The runner only *combines* their verdicts; it never lets text reach the output path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from viscurate.baselines.judges import (
    JudgeVerdict,
    LlmJudge,
    TextJudge,
    TextRecord,
    text_record_from_spec,
)
from viscurate.benchmark.ground_truth import DesignedLabel, GroundTruthGraph
from viscurate.benchmark.metrics import (
    PRF,
    DivergenceRow,
    SafetyStats,
    abstention_rate,
    confusion_matrix,
    divergence_by_true_relation,
    mergeable_prf,
    per_relation_prf,
    precision_on_distinct,
)
from viscurate.benchmark.tracks import Track, Verdict
from viscurate.config import ThresholdConfig
from viscurate.equivalence.backends import (
    PerceptualBackend,
    SemanticBackend,
    cosine_distance,
    ssim_distance,
)
from viscurate.equivalence.calibrate import LabeledPair, ThresholdFit, calibrate_thresholds
from viscurate.equivalence.candidates import candidate_pairs, compute_fingerprints, normalize_pair
from viscurate.equivalence.compare import OutputProvider, mean, quantile, worst_case
from viscurate.equivalence.param_alignment import ParamAlignment
from viscurate.equivalence.relations import Relation
from viscurate.equivalence.taxonomy import classify
from viscurate.skills.canonicalize import max_abs_pixel_diff
from viscurate.skills.model import Params, SkillSpec

__all__ = [
    "BenchmarkResult",
    "CalibrationOutcome",
    "PairMeasurement",
    "PairOutcome",
    "calibrate_from_result",
    "labeled_pairs_for_calibration",
    "measure_pair",
    "run_benchmark",
    "split_pairs_by_cluster",
]

Pair = tuple[str, str]


@dataclass(frozen=True)
class PairMeasurement:
    """The full distance vector for a pair (every level), independent of stop-at-first.

    Calibration needs distances for *every* pair regardless of where the taxonomy stopped, so
    this is computed separately from :func:`classify` (CLAUDE.md §3.5.5). ``nan`` marks a level
    whose backend was absent.
    """

    l_inf: float
    lpips: float
    ssim_dist: float
    dino_p90: float
    dino_mean: float


@dataclass(frozen=True)
class PairOutcome:
    """Everything recorded for one scored pair (for the report and per-pair drill-down)."""

    pair: Pair
    truth: DesignedLabel
    output: Verdict
    text: dict[str, Verdict]
    measurement: PairMeasurement


def _sweep(
    alignment: ParamAlignment | None, provider: OutputProvider, a_id: str, b_id: str
) -> list[tuple[Params, Params]]:
    """The matched sweep for the pair, or the single default binding (mirrors taxonomy)."""
    if alignment is not None:
        ms = alignment.matched_sweep(a_id, b_id)
        if ms:
            return ms
    return [(provider.default_params(a_id), provider.default_params(b_id))]


def measure_pair(
    provider: OutputProvider,
    a_id: str,
    b_id: str,
    *,
    thresholds: ThresholdConfig,
    alignment: ParamAlignment | None = None,
    perceptual: PerceptualBackend | None = None,
    semantic: SemanticBackend | None = None,
    clip: SemanticBackend | None = None,
    seed: int | None = None,
) -> PairMeasurement:
    """Compute worst-case pixel/LPIPS/SSIM and p90/mean DINO over the matched sweep × probes."""
    sweep = _sweep(alignment, provider, a_id, b_id)
    pix: dict[str, float] = {}
    lp: dict[str, float] = {}
    ss: dict[str, float] = {}
    dino: dict[str, float] = {}
    for i, (pa, pb) in enumerate(sweep):
        ao = provider.outputs(a_id, pa, seed=seed)
        bo = provider.outputs(b_id, pb, seed=seed)
        common = ao.common(bo)
        if not common:
            continue
        for p in common:
            pix[f"{i}:{p}"] = max_abs_pixel_diff(ao.canon[p], bo.canon[p])
        if perceptual is not None:
            for p in common:
                lp[f"{i}:{p}"] = perceptual.distance(ao.canon[p].rgb, bo.canon[p].rgb)
                ss[f"{i}:{p}"] = ssim_distance(ao.canon[p].rgb, bo.canon[p].rgb)
        if semantic is not None:
            fa = semantic.features([ao.canon[p].rgb for p in common])
            fb = semantic.features([bo.canon[p].rgb for p in common])
            ca = clip.features([ao.canon[p].rgb for p in common]) if clip is not None else None
            cb = clip.features([bo.canon[p].rgb for p in common]) if clip is not None else None
            for j, p in enumerate(common):
                d = cosine_distance(fa[j], fb[j])
                if ca is not None and cb is not None:
                    d = max(d, cosine_distance(ca[j], cb[j]))
                dino[f"{i}:{p}"] = d
    nan = float("nan")
    return PairMeasurement(
        l_inf=worst_case(pix).value if pix else float("inf"),
        lpips=worst_case(lp).value if lp else nan,
        ssim_dist=worst_case(ss).value if ss else nan,
        dino_p90=quantile(dino, thresholds.semantic_quantile) if dino else nan,
        dino_mean=mean(dino) if dino else nan,
    )


def _output_verdict(
    a_spec: SkillSpec,
    b_spec: SkillSpec,
    provider: OutputProvider,
    *,
    thresholds: ThresholdConfig,
    alignment: ParamAlignment | None,
    perceptual: PerceptualBackend | None,
    semantic: SemanticBackend | None,
    clip: SemanticBackend | None,
    seed: int | None,
) -> Verdict:
    res = classify(
        provider.comparator_view(a_spec.id),
        provider.comparator_view(b_spec.id),
        provider,
        thresholds=thresholds,
        perceptual=perceptual,
        semantic=semantic,
        clip=clip,
        alignment=alignment,
        seed=seed,
    )
    score = res.distances.get("lpips", res.distances.get("l_inf", float("nan")))
    return Verdict(
        relation=res.relation,
        mergeable=res.licenses_merge,
        direction=res.direction,
        score=score,
        uncertain=res.is_uncertain,
        reason=res.reason,
    )


def _to_verdict(jv: JudgeVerdict) -> Verdict:
    return Verdict(relation=jv.relation, mergeable=jv.mergeable, score=jv.similarity)


@dataclass(frozen=True)
class BenchmarkResult:
    """The full output of one benchmark run (predictions + metrics + measurements + manifest)."""

    pairs: tuple[Pair, ...]
    truth: dict[Pair, DesignedLabel]
    output_track: Track
    text_tracks: tuple[Track, ...]
    outcomes: tuple[PairOutcome, ...]
    measurements: dict[Pair, PairMeasurement]
    meta: dict[str, object] = field(default_factory=dict)

    # -- convenience views the report uses ----------------------------------------
    def track(self, name: str) -> Track:
        for t in (self.output_track, *self.text_tracks):
            if t.name == name:
                return t
        raise KeyError(name)

    def per_relation(self, track_name: str) -> dict[Relation, PRF]:
        return per_relation_prf(self.truth, self.track(track_name).predictions)

    def confusion(self, track_name: str) -> dict[Relation, dict[Relation, int]]:
        return confusion_matrix(self.truth, self.track(track_name).predictions)

    def mergeable(self, track_name: str) -> PRF:
        return mergeable_prf(self.truth, self.track(track_name).predictions)

    def safety(self, track_name: str, *, hard_negatives_only: bool = False) -> SafetyStats:
        return precision_on_distinct(
            self.truth, self.track(track_name).predictions, hard_negatives_only=hard_negatives_only
        )

    def abstention(self, track_name: str) -> float:
        return abstention_rate(self.track(track_name).predictions)

    def divergence(self, text_track_name: str) -> list[DivergenceRow]:
        return divergence_by_true_relation(
            self.output_track.predictions, self.track(text_track_name).predictions, self.truth
        )


def run_benchmark(
    specs: Sequence[SkillSpec],
    provider: OutputProvider,
    ground_truth: GroundTruthGraph,
    *,
    thresholds: ThresholdConfig,
    text_judges: Sequence[TextJudge],
    alignment: ParamAlignment | None = None,
    perceptual: PerceptualBackend | None = None,
    semantic: SemanticBackend | None = None,
    clip: SemanticBackend | None = None,
    pairs: set[Pair] | None = None,
    screening_ids: Sequence[str] = (),
    candidate_k: int = 5,
    candidate_max_distance: float = 0.5,
    compute_measurements: bool = True,
    seed: int | None = None,
    meta: dict[str, object] | None = None,
) -> BenchmarkResult:
    """Run every judge track over the candidate pairs and score them against ``G0``.

    Candidate pairs default to output-based generation (CLAUDE.md §3.5.6) **unioned with the
    planted ``G0`` structure**, so the designed relations are always scored even if the
    fingerprint NN misses them. Pass ``pairs`` to score an explicit set instead.
    """
    spec_by_id = {s.id: s for s in specs}
    views = [provider.comparator_view(s.id) for s in specs]

    # -- candidate pairs ----------------------------------------------------------
    if pairs is None:
        if semantic is None:
            raise ValueError("candidate generation needs a semantic backend (or pass `pairs`)")
        fps = compute_fingerprints(
            views, provider, semantic, screening_ids=screening_ids, seed=seed
        )
        generated = candidate_pairs(views, fps, k=candidate_k, max_distance=candidate_max_distance)
    else:
        generated = {normalize_pair(a, b) for a, b in pairs}
    valid = set(spec_by_id)
    designed = {p for p in ground_truth.designed_pairs() if p[0] in valid and p[1] in valid}
    scored = sorted(generated | designed)

    # -- text records (the one place text is read) --------------------------------
    records: dict[str, TextRecord] = {s.id: text_record_from_spec(s) for s in specs}

    # -- per-pair verdicts --------------------------------------------------------
    truth: dict[Pair, DesignedLabel] = {}
    out_preds: dict[Pair, Verdict] = {}
    text_preds: dict[str, dict[Pair, Verdict]] = {j.name: {} for j in text_judges}
    measurements: dict[Pair, PairMeasurement] = {}
    outcomes: list[PairOutcome] = []
    runnable = [j for j in text_judges if not (isinstance(j, LlmJudge) and not j.available)]

    for a_id, b_id in scored:
        a_spec, b_spec = spec_by_id[a_id], spec_by_id[b_id]
        truth[(a_id, b_id)] = ground_truth.label(a_id, b_id)
        ov = _output_verdict(
            a_spec,
            b_spec,
            provider,
            thresholds=thresholds,
            alignment=alignment,
            perceptual=perceptual,
            semantic=semantic,
            clip=clip,
            seed=seed,
        )
        out_preds[(a_id, b_id)] = ov
        tv: dict[str, Verdict] = {}
        for judge in runnable:
            v = _to_verdict(judge.verdict(records[a_id], records[b_id]))
            text_preds[judge.name][(a_id, b_id)] = v
            tv[judge.name] = v
        m = (
            measure_pair(
                provider,
                a_id,
                b_id,
                thresholds=thresholds,
                alignment=alignment,
                perceptual=perceptual,
                semantic=semantic,
                clip=clip,
                seed=seed,
            )
            if compute_measurements
            else PairMeasurement(
                float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
            )
        )
        measurements[(a_id, b_id)] = m
        outcomes.append(PairOutcome((a_id, b_id), truth[(a_id, b_id)], ov, tv, m))

    # -- assemble tracks ----------------------------------------------------------
    output_track = Track(name="output-grounded", kind="output", predictions=out_preds)
    text_tracks: list[Track] = []
    for judge in text_judges:
        ran = judge in runnable
        text_tracks.append(
            Track(
                name=judge.name,
                kind="text",
                predictions=text_preds[judge.name],
                ran=ran,
                note="" if ran else "no LLM client configured — track not run",
            )
        )

    full_meta: dict[str, object] = {
        "n_skills": len(specs),
        "n_pairs": len(scored),
        "seed": seed,
        "perceptual_backend": getattr(perceptual, "name", None),
        "semantic_backend": getattr(semantic, "name", None),
        "clip_backend": getattr(clip, "name", None),
        "thresholds_calibrated": thresholds.calibrated,
        **(meta or {}),
    }
    return BenchmarkResult(
        pairs=tuple(scored),
        truth=truth,
        output_track=output_track,
        text_tracks=tuple(text_tracks),
        outcomes=tuple(outcomes),
        measurements=measurements,
        meta=full_meta,
    )


def labeled_pairs_for_calibration(result: BenchmarkResult) -> list[LabeledPair]:
    """Turn a run's measurements + true labels into calibration inputs (CLAUDE.md §3.5.5).

    Only finite distances are included, so a pair missing a level (no backend) does not poison
    that threshold's fit.
    """
    import math

    out: list[LabeledPair] = []
    for pair, m in result.measurements.items():
        dist: dict[str, float] = {}
        if not math.isnan(m.lpips) and not math.isinf(m.lpips):
            dist["lpips"] = m.lpips
        if not math.isnan(m.dino_p90) and not math.isinf(m.dino_p90):
            dist["dino_p90"] = m.dino_p90
        if dist:
            out.append(LabeledPair(true=result.truth[pair].relation, distances=dist))
    return out


@dataclass(frozen=True)
class CalibrationOutcome:
    """The result of calibrating thresholds on the cluster-disjoint split of one run."""

    config: ThresholdConfig
    fits: dict[str, ThresholdFit]
    n_calibration: int
    n_test: int


def calibrate_from_result(
    result: BenchmarkResult,
    spec_by_id: dict[str, SkillSpec],
    *,
    calibration_families: set[str],
    base: ThresholdConfig,
    date: str,
    min_precision: float = 0.99,
    min_recall: float = 0.5,
) -> CalibrationOutcome:
    """Calibrate τ_perc/τ_sem/δ on the **calibration** cluster, frozen apart from test.

    The split is cluster-disjoint (CLAUDE.md §2.5) so calibration never leaks into the test
    metrics. The returned config carries the provenance stamp (split hash + date) required by
    the config validator before any metric may be reported (CLAUDE.md §3.5.5).
    """
    calib, test = split_pairs_by_cluster(
        result, spec_by_id, calibration_families=calibration_families
    )
    config, fits = calibrate_thresholds(
        calib, base=base, date=date, min_precision=min_precision, min_recall=min_recall
    )
    return CalibrationOutcome(config=config, fits=fits, n_calibration=len(calib), n_test=len(test))


def split_pairs_by_cluster(
    result: BenchmarkResult,
    spec_by_id: dict[str, SkillSpec],
    *,
    calibration_families: set[str],
) -> tuple[list[LabeledPair], list[LabeledPair]]:
    """Cluster-disjoint split (CLAUDE.md §2.5): a family is wholly in calibration or in test.

    A pair lands in *calibration* iff both endpoints' families are in ``calibration_families``,
    in *test* iff both are outside it; cross-cluster pairs are dropped from both so no skill (or
    family) leaks across the split. Returns ``(calibration_pairs, test_pairs)``.
    """
    import math

    calib: list[LabeledPair] = []
    test: list[LabeledPair] = []
    for (a_id, b_id), m in result.measurements.items():
        fam_a = spec_by_id[a_id].metadata.family
        fam_b = spec_by_id[b_id].metadata.family
        in_a = fam_a in calibration_families
        in_b = fam_b in calibration_families
        if in_a != in_b:
            continue  # cross-cluster → excluded to prevent leakage
        dist: dict[str, float] = {}
        if not math.isnan(m.lpips) and not math.isinf(m.lpips):
            dist["lpips"] = m.lpips
        if not math.isnan(m.dino_p90) and not math.isinf(m.dino_p90):
            dist["dino_p90"] = m.dino_p90
        if not dist:
            continue
        lp = LabeledPair(true=result.truth[(a_id, b_id)].relation, distances=dist)
        (calib if in_a else test).append(lp)
    return calib, test

"""Phase 4 — the equivalence benchmark (CLAUDE.md §3.5.9).

Covers the answer key (``G0`` validation + lookups), the metrics (per-relation P/R/F1,
mergeable decision, safety numbers, divergence), the human-review/κ infrastructure, and the
end-to-end runner with deterministic *fake* backends — the same no-ML discipline as the Phase-3
suite. The real LPIPS/DINO/CLIP run is a separate, GPU-targeted step (CLI ``run-benchmark``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from viscurate.baselines.judges import (
    EmbeddingCosineJudge,
    LlmJudge,
    NameMatchJudge,
    TfidfEmbedder,
    text_record_from_spec,
)
from viscurate.benchmark import (
    DesignedLabel,
    Verdict,
    abstention_rate,
    calibrate_from_result,
    cohen_kappa,
    divergence_by_true_relation,
    extract_review_slice,
    fleiss_kappa,
    inter_annotator_agreement,
    labeled_pairs_for_calibration,
    load_ground_truth,
    load_review_labels,
    mergeable_prf,
    per_relation_prf,
    precision_on_distinct,
    run_benchmark,
    split_pairs_by_cluster,
    write_report,
    write_review_template,
)
from viscurate.benchmark.ground_truth import GroundTruthGraph, GroundTruthSpec
from viscurate.config import ThresholdConfig
from viscurate.equivalence.candidates import normalize_pair
from viscurate.equivalence.compare import BatteryEvaluator
from viscurate.equivalence.param_alignment import load_param_alignment
from viscurate.equivalence.relations import Direction, Relation
from viscurate.skills.library import build_builtin_registry

G0_PATH = "configs/ground_truth_g0.yaml"
ALIGN_PATH = "configs/param_alignment.yaml"


# --------------------------------------------------------------------------------------------
# Fake backends (no torch), reused from the Phase-3 pattern.
# --------------------------------------------------------------------------------------------


class FakePerceptual:
    name = "fake-perc"

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float("inf") if a.shape != b.shape else float(np.mean(np.abs(a - b)))


class FakeSemantic:
    name = "fake-sem"

    def features(self, imgs: Sequence[np.ndarray]) -> np.ndarray:
        if not imgs:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in imgs])


# --------------------------------------------------------------------------------------------
# G0 — the designed relation graph (answer key).
# --------------------------------------------------------------------------------------------


def test_g0_loads_and_validates_against_the_real_library() -> None:
    reg = build_builtin_registry()
    g0 = load_ground_truth(G0_PATH, valid_ids={s.id for s in reg.all()})
    assert len(g0.designed_pairs()) == 24


def test_g0_lookups_and_directions() -> None:
    g0 = load_ground_truth(G0_PATH)
    blur = g0.label("blur_gaussian_v1", "blur_box_v1")
    assert blur.relation is Relation.DISTINCT and blur.is_hard_negative
    assert g0.label("rotate_90_v1", "rotate_canvas_degrees_v1").direction is Direction.B_SUBSUMES_A
    assert g0.label("rotate_canvas_degrees_v1", "rotate_90_v1").direction is Direction.A_SUBSUMES_B
    assert (
        g0.label("grayscale_bt601_v1", "grayscale_bt709_v1").relation
        is Relation.SEMANTIC_PRESERVING
    )
    assert (
        g0.label("morphology_dilate_v1", "morphology_erode_v1").relation is Relation.COMPLEMENTARY
    )
    assert g0.label("swirl_distort_v1", "sepia_tone_v1").relation is Relation.DISTINCT  # default


def test_g0_rejects_subsumption_cycle() -> None:
    spec = GroundTruthSpec.model_validate(
        {
            "subsumption": [
                {"spec": "a", "gen": "b"},
                {"spec": "b", "gen": "c"},
                {"spec": "c", "gen": "a"},
            ]
        }
    )
    with pytest.raises(ValueError, match="not a DAG"):
        GroundTruthGraph(spec)


def test_g0_rejects_duplicate_pair() -> None:
    spec = GroundTruthSpec.model_validate(
        {"semantic_preserving": [["a", "b"], ["b", "a"]]}  # same unordered pair twice
    )
    with pytest.raises(ValueError, match="more than once"):
        GroundTruthGraph(spec)


def test_g0_rejects_non_transitive_exact() -> None:
    spec = GroundTruthSpec.model_validate({"exact": [["a", "b"], ["b", "c"]]})  # missing (a, c)
    with pytest.raises(ValueError, match="transitively closed"):
        GroundTruthGraph(spec)


def test_g0_rejects_unknown_ids() -> None:
    spec = GroundTruthSpec.model_validate({"semantic_preserving": [["ghost_v1", "phantom_v1"]]})
    with pytest.raises(ValueError, match="unknown skill ids"):
        GroundTruthGraph(spec, valid_ids={"real_v1"})


# --------------------------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------------------------


def _truth() -> dict[tuple[str, str], DesignedLabel]:
    return {
        ("a", "b"): DesignedLabel(Relation.EXACT),
        ("c", "d"): DesignedLabel(Relation.DISTINCT, is_hard_negative=True),
        ("e", "f"): DesignedLabel(Relation.SEMANTIC_PRESERVING),
        ("g", "h"): DesignedLabel(Relation.DISTINCT),
    }


def _text_preds() -> dict[tuple[str, str], Verdict]:
    return {
        ("a", "b"): Verdict(Relation.EXACT, mergeable=True),
        ("c", "d"): Verdict(Relation.EXACT, mergeable=True),  # false merge on a hard negative
        ("e", "f"): Verdict(Relation.DISTINCT, mergeable=False),
        ("g", "h"): Verdict(Relation.DISTINCT, mergeable=False),
    }


def test_mergeable_prf_counts_false_merge() -> None:
    prf = mergeable_prf(_truth(), _text_preds())
    assert prf.tp == 1 and prf.fp == 1 and prf.fn == 0
    assert prf.precision == pytest.approx(0.5) and prf.recall == pytest.approx(1.0)


def test_precision_on_distinct_and_hard_negative_slice() -> None:
    alld = precision_on_distinct(_truth(), _text_preds())
    assert alld.n_distinct == 2 and alld.false_merges == 1
    assert alld.false_merge_rate == pytest.approx(0.5)
    hard = precision_on_distinct(_truth(), _text_preds(), hard_negatives_only=True)
    assert hard.n_distinct == 1 and hard.false_merges == 1
    assert hard.precision_on_distinct == pytest.approx(0.0)


def test_per_relation_prf_and_abstention() -> None:
    out = {
        ("a", "b"): Verdict(Relation.EXACT, mergeable=True),
        ("c", "d"): Verdict(Relation.DISTINCT, mergeable=False),
        ("e", "f"): Verdict(Relation.UNCERTAIN, mergeable=False, uncertain=True),
        ("g", "h"): Verdict(Relation.DISTINCT, mergeable=False),
    }
    prf = per_relation_prf(_truth(), out)
    assert prf[Relation.EXACT].precision == pytest.approx(1.0)
    assert prf[Relation.SEMANTIC_PRESERVING].recall == pytest.approx(0.0)  # abstained
    assert abstention_rate(out) == pytest.approx(0.25)


def test_divergence_separates_over_and_under_merge() -> None:
    output = {
        ("c", "d"): Verdict(Relation.DISTINCT, mergeable=False),  # output keeps separate
        ("a", "b"): Verdict(Relation.EXACT, mergeable=True),  # output finds redundancy
    }
    text = {
        ("c", "d"): Verdict(Relation.EXACT, mergeable=True),  # text over-merges
        ("a", "b"): Verdict(Relation.DISTINCT, mergeable=False),  # text under-merges
    }
    truth = {
        ("c", "d"): DesignedLabel(Relation.DISTINCT, is_hard_negative=True),
        ("a", "b"): DesignedLabel(Relation.EXACT),
    }
    rows = {r.slice_name: r for r in divergence_by_true_relation(output, text, truth)}
    assert rows["ALL"].disagree == 2
    assert rows["ALL"].text_over_merge == 1 and rows["ALL"].text_under_merge == 1
    assert rows["hard_negative"].text_over_merge == 1


# --------------------------------------------------------------------------------------------
# Human review + κ.
# --------------------------------------------------------------------------------------------


def test_cohen_kappa_known_value_and_perfect() -> None:
    a = [Relation.EXACT, Relation.EXACT, Relation.DISTINCT, Relation.SEMANTIC_PRESERVING]
    b = [Relation.EXACT, Relation.DISTINCT, Relation.DISTINCT, Relation.SEMANTIC_PRESERVING]
    assert cohen_kappa(a, b) == pytest.approx(0.6363636, abs=1e-5)
    assert cohen_kappa(a, a) == pytest.approx(1.0)


def test_fleiss_kappa_perfect_agreement() -> None:
    ratings = [
        [Relation.EXACT, Relation.EXACT, Relation.EXACT],
        [Relation.DISTINCT, Relation.DISTINCT, Relation.DISTINCT],
    ]
    assert fleiss_kappa(ratings) == pytest.approx(1.0)


def test_inter_annotator_agreement_pending_then_computed() -> None:
    assert inter_annotator_agreement({}).status == "pending"
    one = {("a", "b"): [Relation.EXACT]}
    assert inter_annotator_agreement(one).status == "pending"  # needs ≥2 annotators
    two = {
        ("a", "b"): [Relation.EXACT, Relation.EXACT],
        ("c", "d"): [Relation.DISTINCT, Relation.SEMANTIC_PRESERVING],
    }
    res = inter_annotator_agreement(two)
    assert res.status == "computed" and res.n_annotators == 2 and res.kappa is not None


# --------------------------------------------------------------------------------------------
# End-to-end runner (fake backends, tiny battery) + calibration + review + report.
# --------------------------------------------------------------------------------------------


def _small_run():  # type: ignore[no-untyped-def]
    reg = build_builtin_registry()
    ids = [
        "blur_gaussian_v1",
        "blur_box_v1",
        "grayscale_bt601_v1",
        "grayscale_bt709_v1",
        "rotate_90_v1",
        "rotate_canvas_degrees_v1",
        "morphology_dilate_v1",
        "morphology_erode_v1",
        "invert_v1",
        "sepia_tone_v1",
    ]
    skills = [reg.get(i) for i in ids]
    specs = [s.to_spec() for s in skills]
    rng = np.random.default_rng(0)
    battery = [
        ("tex", rng.integers(20, 200, (48, 48, 3), dtype=np.uint8)),
        ("grad", np.tile(np.linspace(20, 200, 48, dtype=np.uint8)[None, :, None], (48, 1, 3))),
    ]
    provider = BatteryEvaluator(skills, battery, seed=0)
    g0 = load_ground_truth(G0_PATH, valid_ids=None)  # scoring a subset
    align = load_param_alignment(ALIGN_PATH)
    emb = TfidfEmbedder([text_record_from_spec(s).text() for s in specs])
    judges = [NameMatchJudge(), EmbeddingCosineJudge(emb, tau=0.3), LlmJudge()]
    result = run_benchmark(
        specs,
        provider,
        g0,
        thresholds=ThresholdConfig(),
        text_judges=judges,
        alignment=align,
        perceptual=FakePerceptual(),
        semantic=FakeSemantic(),
        screening_ids=["tex", "grad"],
        seed=0,
    )
    return result, specs


def test_runner_produces_the_headline_divergence() -> None:
    result, _ = _small_run()
    assert result.meta["n_pairs"] > 0
    blur = normalize_pair("blur_gaussian_v1", "blur_box_v1")
    # The safety property: the output verifier does NOT license the blur merge…
    assert not result.output_track.predictions[blur].mergeable
    # …while the description-embedding strawman does (similar text) — the divergence.
    assert result.track("embedding-cosine").predictions[blur].mergeable
    rows = {r.slice_name: r for r in result.divergence("embedding-cosine")}
    assert rows["hard_negative"].text_over_merge >= 1


def test_runner_marks_unavailable_llm_track_not_run() -> None:
    result, _ = _small_run()
    llm = result.track("llm-on-descriptions")
    assert llm.ran is False and "no LLM" in llm.note
    assert llm.predictions == {}


def test_calibration_from_run_stamps_provenance() -> None:
    result, specs = _small_run()
    pairs = labeled_pairs_for_calibration(result)
    assert pairs  # finite distances present
    spec_by_id = {s.id: s for s in specs}
    calib, test = split_pairs_by_cluster(
        result, spec_by_id, calibration_families={"blur", "geometric"}
    )
    assert calib and test  # cluster-disjoint, both non-empty
    outcome = calibrate_from_result(
        result,
        spec_by_id,
        calibration_families={"blur", "geometric"},
        base=ThresholdConfig(),
        date="2026-06-24",
    )
    assert outcome.config.calibrated is True
    assert outcome.config.calibration_date == "2026-06-24"
    assert (
        outcome.config.calibration_split_hash and len(outcome.config.calibration_split_hash) == 64
    )


def test_review_slice_export_and_kappa_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result, specs = _small_run()
    spec_by_id = {s.id: s for s in specs}
    items = extract_review_slice(result, spec_by_id)
    pairs = {it.pair for it in items}
    assert normalize_pair("grayscale_bt601_v1", "grayscale_bt709_v1") in pairs  # SEMANTIC
    assert normalize_pair("rotate_90_v1", "rotate_canvas_degrees_v1") in pairs  # SUBSUMPTION

    # Two annotators "fill" the template (here programmatically) → κ computes.
    tmpl = write_review_template(items, tmp_path / "review.json")
    import json

    base = json.loads(tmpl.read_text())
    for ann, rel in (("a", Relation.SEMANTIC_PRESERVING), ("b", Relation.SEMANTIC_PRESERVING)):
        for it in base["items"]:
            it["label"] = rel.value
        (tmp_path / f"ann_{ann}.json").write_text(json.dumps(base))
    labels = load_review_labels([tmp_path / "ann_a.json", tmp_path / "ann_b.json"])
    res = inter_annotator_agreement(labels)
    assert res.status == "computed" and res.kappa == pytest.approx(1.0)  # both said SEMANTIC


def test_write_report_emits_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result, specs = _small_run()
    paths = write_report(result, tmp_path / "out", specs=specs)
    for key in ("report", "divergence_csv", "pairs_csv", "manifest", "review_template"):
        assert paths[key].exists()
    assert "Divergence" in paths["report"].read_text()
    import json

    manifest = json.loads(paths["manifest"].read_text())
    assert (
        manifest["phase"] == 4 and "canon_version" not in manifest
    )  # canon added only by CLI meta

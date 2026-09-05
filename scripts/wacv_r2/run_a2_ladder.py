#!/usr/bin/env python
"""A2 — the baseline ladder, evaluated fairly at matched operating points.

Eight rungs. 1/2/6 and 8 already existed; 3/4/5/7 are the new ones (A2/01 §2). The point of this
script is not only to add rungs but to fix how ALL of them are compared:

FIX 1 — sweep every threshold-based rung, including the two already in the paper.
    `name-match` ships pinned at tau=0.5 and TF-IDF at tau=0.6. Comparing those fixed points
    against optimized new rungs is unfair in our own favour (A2/01 §4.2). Every threshold rung is
    swept over its full score range here, and each reports AUPRC (threshold-free) plus a matched
    operating point derived by one identical rule.

FIX 2 — report the false-discovery rate on the MERGE decision, not only precision on
    non-equivalence. With 926 distinct vs ~18 non-distinct pairs, predicting DISTINCT for
    everything already scores 0.981 precision-on-non-equivalence, so that number cannot carry a
    safety claim (A2/01 §4.1). FDR_merge = FP / (TP + FP) among predicted-mergeable is the
    quantity that actually governs silent merges, and it is reported first.

FIX 3 — unparseable LLM replies are ABSTAIN, never DISTINCT.
    The shipped LlmJudge maps an unparseable reply to DISTINCT, which folds model failure into a
    safety number: the baseline looks safest exactly when it fails hardest. Abstentions are
    counted separately and the abstention rate is reported alongside (A2/01 §3 rung 7a).

FIX 4 — every rate carries raw counts and an exact Clopper-Pearson interval, and the LLM rungs
    are repeated so dispersion is measured rather than assumed (A2/01 §3 rung 7b).

The positive class is MERGEABLE (EXACT u PERCEPTUAL) throughout, because that is the decision all
eight rungs can express and the one whose errors are destructive.

NOTE ON `G_0` SUPPORT: `G_0` designates no EXACT and no PERCEPTUAL pair, so on `G_0` the mergeable
positive class is EMPTY. Precision/recall/F1/AUPRC for the positive class are therefore undefined
there and are reported as such rather than as 0.000 (which is what the shipped report prints).
This is why A2/01 §5 calls `G_rho` the only set with EXACT/PERCEPTUAL support, and why the ladder
must be run on `G_rho` to say anything about merge recall at all.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from viscurate.baselines.code_judges import (
    SEMANTICS_PRESERVING,
    STRUCTURE_ONLY,
    AstCloneJudge,
    CodeRecord,
    _strip_docstring,
)
from viscurate.baselines.judges import (
    EmbeddingCosineJudge,
    JudgeVerdict,
    NameMatchJudge,
    OpenAIClient,
    TextRecord,
    TfidfEmbedder,
    text_record_from_spec,
)
from viscurate.equivalence.relations import Relation
from viscurate.skills.library import build_builtin_registry

MERGEABLE = (Relation.EXACT, Relation.PERCEPTUAL)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact TWO-SIDED 95% binomial interval (alpha/2 in each tail)."""
    from scipy.stats import beta

    if n == 0:
        return (0.0, 1.0)
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (low, high)


def clopper_pearson_upper_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact ONE-SIDED 95% upper bound — the quantity a "rate is at most X" claim needs.

    Both bounds are reported because they differ materially at these counts and are easy to
    conflate: for 0/6 the one-sided 95% upper bound is 0.393 (the figure the revision plan
    quotes) while the two-sided 95% interval's upper end is 0.459. A safety claim of the form
    "the false-merge rate is below X" is a one-sided statement, so the one-sided bound is the
    honest companion to it.
    """
    from scipy.stats import beta

    if n == 0:
        return 1.0
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


# ==============================================================================================
# Units under test
# ==============================================================================================
@dataclass
class Unit:
    """One skill as each rung is allowed to see it."""

    id: str
    text: TextRecord
    code: CodeRecord | None
    excluded_reason: str = ""


def build_units() -> tuple[dict[str, Unit], list[dict[str, str]]]:
    """Text + source surface for every built-in skill.

    The code surface is the implementation PLUS the skill's declared parameter defaults, rendered
    as a leading assignment. Without the defaults, corruption defect type (v) — parameter-default
    drift — is invisible to every code rung *by construction*, because the drifted skill shares
    the original's function body byte for byte and only its declared defaults differ. Including
    them is also what a curator reading the skill actually sees, and it is exactly the asymmetry
    A2/01 §3 rung 5 requires both AST variants to be tested against.
    """
    registry = build_builtin_registry()
    units: dict[str, Unit] = {}
    exclusions: list[dict[str, str]] = []
    for skill in registry.all():
        text = text_record_from_spec(skill.to_spec())
        try:
            body = inspect.getsource(skill.fn)
        except (OSError, TypeError) as exc:
            exclusions.append(
                {"unit": skill.id, "stage": "code_record", "reason": f"{type(exc).__name__}: {exc}"}
            )
            units[skill.id] = Unit(skill.id, text, None, f"source unavailable: {exc}")
            continue
        defaults = skill.params_schema.defaults()
        source = f"_DEFAULTS = {defaults!r}\n" + inspect.cleandoc(body)
        units[skill.id] = Unit(
            skill.id,
            text,
            CodeRecord(id=skill.id, source=source, source_no_doc=_strip_docstring(source)),
        )
    return units, exclusions


# ==============================================================================================
# Metrics
# ==============================================================================================
def score_at_threshold(
    scores: list[float], truth_mergeable: list[bool], hard_neg: list[bool], tau: float
) -> dict[str, Any]:
    """One operating point of a threshold rung. Predicted mergeable iff score >= tau."""
    tp = fp = fn = tn = 0
    fm_hard = 0
    n_hard = sum(hard_neg)
    for s, t, h in zip(scores, truth_mergeable, hard_neg, strict=True):
        pred = s >= tau
        if pred and t:
            tp += 1
        elif pred and not t:
            fp += 1
            if h:
                fm_hard += 1
        elif not pred and t:
            fn += 1
        else:
            tn += 1
    n_pred_merge = tp + fp
    n_distinct = fp + tn
    return {
        "tau": tau,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        # THE safety number: of everything we would merge, what fraction is wrong.
        "fdr_merge": (fp / n_pred_merge) if n_pred_merge else None,
        # the paper's current headline, kept for continuity
        "fpr_distinct": (fp / n_distinct) if n_distinct else None,
        # the current CALIBRATION target; inflated by class imbalance, kept for continuity
        "precision_non_equiv": (tn / (tn + fn)) if (tn + fn) else None,
        "recall_merge": (tp / (tp + fn)) if (tp + fn) else None,
        "precision_merge": (tp / n_pred_merge) if n_pred_merge else None,
        "fm_hard": fm_hard,
        "n_hard": n_hard,
    }


def auprc(scores: list[float], truth: list[bool]) -> float | None:
    """Average precision for the mergeable positive class; None when the class is empty."""
    if not any(truth):
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.array(truth, dtype=int), np.array(scores)))


def matched_operating_point(
    scores: list[float],
    truth: list[bool],
    hard_neg: list[bool],
    *,
    precision_floor: float = 0.99,
    recall_floor: float = 0.5,
) -> dict[str, Any]:
    """The single matched rule applied identically to every threshold rung.

    "The most permissive threshold holding precision-on-non-equivalence >= 0.99 subject to
    recall >= 0.5" (A2/01 §4 rule 2). Most permissive = lowest tau, since lower tau merges more.
    When no threshold satisfies both, the constraint that failed is recorded rather than silently
    relaxed, and the best-achievable point under the precision floor alone is returned.
    """
    if not any(truth):
        # No mergeable positive exists, so precision-on-non-equivalence is 1.0 at EVERY threshold
        # (there can be no false negative) and the recall floor is unsatisfiable. The rule would
        # therefore select the most permissive threshold and merge everything. That is not a
        # matched operating point, it is the class-imbalance pathology this rule exists to avoid,
        # so the rule is declared inapplicable instead of returning a degenerate answer.
        row = score_at_threshold(scores, truth, hard_neg, 1.0)
        row["matched_status"] = "inapplicable_no_positive_support"
        return row
    candidates = sorted({round(s, 6) for s in scores} | {0.0, 1.0})
    feasible: list[dict[str, Any]] = []
    precision_only: list[dict[str, Any]] = []
    for tau in candidates:
        row = score_at_threshold(scores, truth, hard_neg, tau)
        p_ne = row["precision_non_equiv"]
        r = row["recall_merge"]
        if p_ne is not None and p_ne >= precision_floor:
            precision_only.append(row)
            if r is not None and r >= recall_floor:
                feasible.append(row)
    if feasible:
        best = min(feasible, key=lambda r: r["tau"])
        best["matched_status"] = "both_constraints_met"
        return best
    if precision_only:
        best = min(precision_only, key=lambda r: r["tau"])
        best["matched_status"] = (
            "recall_floor_unreachable"
            if any(t for t in truth)
            else "recall_undefined_no_positive_support"
        )
        return best
    row = score_at_threshold(scores, truth, hard_neg, 1.0)
    row["matched_status"] = "precision_floor_unreachable"
    return row


# ==============================================================================================
# Main
# ==============================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-csv", required=True, help="Stage-1 pairs.csv (defines the pair set)")
    ap.add_argument("--out", default="results/wacv_r2/a2_ladder")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--sentence-model", default="sentence-transformers/all-mpnet-base-v2", help="rung 3"
    )
    ap.add_argument("--code-model", default="microsoft/unixcoder-base", help="rung 4")
    ap.add_argument("--llm-source-model", default="", help="rung 7; empty = skip")
    ap.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    ap.add_argument("--llm-repeats", type=int, default=3)
    ap.add_argument("--llm-max-pairs", type=int, default=0, help="0 = all; else cap and LOG it")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with Path(args.pairs_csv).open(encoding="utf-8") as _fh:
        rows = list(csv.DictReader(_fh))
    print(f"pair set: {len(rows)} pairs from {args.pairs_csv}")
    units, exclusions = build_units()

    pairs: list[tuple[str, str]] = []
    truth_mergeable: list[bool] = []
    hard_neg: list[bool] = []
    for r in rows:
        pairs.append((r["a"], r["b"]))
        truth_mergeable.append(r["true_relation"] in {m.value for m in MERGEABLE})
        hard_neg.append(r["is_hard_negative"] in ("1", "True", "true"))
    n_pos = sum(truth_mergeable)
    print(f"  mergeable positives in truth: {n_pos} | hard negatives: {sum(hard_neg)}")

    results: dict[str, dict[str, Any]] = {}
    per_pair: dict[str, list[float]] = {}

    def add_threshold_rung(name: str, scorer: Any, default_tau: float, note: str = "") -> None:
        """Score every pair, then report BOTH operating points.

        ``default`` is the rung's shipped threshold — the point the paper's existing numbers were
        produced at, kept so the new table is comparable with Table 2. ``matched`` is the fair,
        identically-derived point, which is only defined when the pair set contains mergeable
        positives. Reporting both is what makes the "we swept the old baselines too" claim
        checkable rather than asserted.
        """
        scores: list[float] = []
        for a, b in pairs:
            ua, ub = units[a], units[b]
            scores.append(scorer(ua, ub))
        per_pair[name] = scores
        default_point = score_at_threshold(scores, truth_mergeable, hard_neg, default_tau)
        default_point["matched_status"] = "shipped_default"
        results[name] = {
            "kind": "threshold",
            "note": note,
            "auprc": auprc(scores, truth_mergeable),
            "default_tau": default_tau,
            "default": default_point,
            "matched": matched_operating_point(scores, truth_mergeable, hard_neg),
            "n_scored": len(scores),
            "score_min": min(scores),
            "score_max": max(scores),
        }
        d = default_point
        m = results[name]["matched"]
        print(
            f"  {name:30} AUPRC={results[name]['auprc']} | default τ={default_tau:.3f}: "
            f"FDR={d['fdr_merge']} fm_hard={d['fm_hard']}/{d['n_hard']} "
            f"| matched: {m['matched_status']}"
        )

    # -- rung 1: name-match ---------------------------------------------------------------------
    nm = NameMatchJudge()
    add_threshold_rung(
        "1-name-match",
        lambda a, b: nm.verdict(a.text, b.text).similarity,
        NameMatchJudge().tau,
        "shipped at τ=0.5; now also swept",
    )

    # -- rung 2: TF-IDF description cosine ------------------------------------------------------
    corpus = [u.text.text() for u in units.values()]
    tfidf = EmbeddingCosineJudge(embedder=TfidfEmbedder(corpus))
    add_threshold_rung(
        "2-tfidf-cosine",
        lambda a, b: tfidf.verdict(a.text, b.text).similarity,
        tfidf.tau,
        "shipped at τ=0.6; now also swept",
    )

    # -- rung 3: sentence embedding -------------------------------------------------------------
    try:
        from viscurate.baselines.embedders import SentenceTransformerEmbedder

        st = SentenceTransformerEmbedder(model_id=args.sentence_model, device=args.device)
        st_judge = EmbeddingCosineJudge(embedder=st)
        add_threshold_rung(
            "3-sentence-embedding",
            lambda a, b: st_judge.verdict(a.text, b.text).similarity,
            st_judge.tau,
            f"model={args.sentence_model}",
        )
        st.close()
    except Exception as exc:
        results["3-sentence-embedding"] = {
            "kind": "threshold",
            "not_run": f"{type(exc).__name__}: {exc}"[:200],
        }
        print(f"  3-sentence-embedding NOT RUN: {exc}")

    # -- rung 4: code embedding (docstrings stripped) -------------------------------------------
    code_ok = [u for u in units.values() if u.code is not None]
    print(f"  code surface available for {len(code_ok)}/{len(units)} skills")
    try:
        from viscurate.baselines.code_judges import TransformerCodeEmbedder

        emb = TransformerCodeEmbedder(model_id=args.code_model, device=args.device)
        cej = CodeEmbeddingScorer(emb)
        add_threshold_rung(
            "4-code-embedding", cej, 0.9, f"model={args.code_model}; docstrings stripped"
        )
        emb.close()
    except Exception as exc:
        results["4-code-embedding"] = {
            "kind": "threshold",
            "not_run": f"{type(exc).__name__}: {exc}"[:200],
        }
        print(f"  4-code-embedding NOT RUN: {exc}")

    # -- rung 5: AST clone, BOTH variants -------------------------------------------------------
    for variant, label in (
        (STRUCTURE_ONLY, "5a-ast-structure-only"),
        (SEMANTICS_PRESERVING, "5b-ast-semantics-preserving"),
    ):
        judge = AstCloneJudge(variant=variant)

        def scorer(a: Unit, b: Unit, _j: Any = judge) -> float:
            if a.code is None or b.code is None:
                return 0.0
            return _j.verdict(a.code, b.code).similarity

        add_threshold_rung(label, scorer, judge.tau, f"variant={variant}")

    # -- rung 8: output-grounded, read from the Stage-1 run --------------------------------------
    og_pred = [r["output_mergeable"] in ("1", "True", "true") for r in rows]
    og_uncertain = [r["output_relation"] == Relation.UNCERTAIN.value for r in rows]
    results["8-output-grounded"] = categorical_summary(
        og_pred, truth_mergeable, hard_neg, abstain=og_uncertain, note="calibrated; from Stage 1"
    )
    print(f"  8-output-grounded  {json.dumps(results['8-output-grounded']['point'])}")

    # -- rung 6: LLM on descriptions, read from the Stage-1 run ---------------------------------
    if "llm-on-descriptions_mergeable" in rows[0]:
        llm_pred = [r["llm-on-descriptions_mergeable"] in ("1", "True", "true") for r in rows]
        results["6-llm-on-descriptions"] = categorical_summary(
            llm_pred, truth_mergeable, hard_neg, note="from Stage 1's judge track"
        )
        print(f"  6-llm-on-descriptions {json.dumps(results['6-llm-on-descriptions']['point'])}")

    # -- rung 7: LLM on source, repeated -------------------------------------------------------
    if args.llm_source_model:
        results["7-llm-on-source"] = run_llm_source(
            pairs, units, truth_mergeable, hard_neg, args, out_dir
        )

    manifest = {
        "artifact": "a2_baseline_ladder",
        "pairs_csv": args.pairs_csv,
        "pairs_csv_sha256": hashlib.sha256(Path(args.pairs_csv).read_bytes()).hexdigest(),
        "n_pairs": len(rows),
        "n_mergeable_positives": n_pos,
        "n_hard_negatives": sum(hard_neg),
        "positive_class": "mergeable = EXACT u PERCEPTUAL",
        "matched_rule": "lowest tau with precision_non_equiv >= 0.99 and recall_merge >= 0.5",
        "sentence_model": args.sentence_model,
        "code_model": args.code_model,
        "llm_source_model": args.llm_source_model,
        "llm_repeats": args.llm_repeats,
        "llm_max_pairs": args.llm_max_pairs,
        "n_exclusions": len(exclusions),
        # "Locked environment" is one of the plan's eight required artifacts, and this ladder is
        # version-sensitive in a way the synthetic study is not: rungs 3/4 are neural encoders and
        # rung 7 is a hosted model, so a metric is only reproducible alongside the versions that
        # produced it.
        "library_versions": _library_versions(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "rungs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "exclusions.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        names = sorted(per_pair)
        w = csv.writer(fh)
        w.writerow(["a", "b", "true_relation", "is_hard_negative", *names])
        for i, (a, b) in enumerate(pairs):
            w.writerow(
                [a, b, rows[i]["true_relation"], rows[i]["is_hard_negative"]]
                + [f"{per_pair[n][i]:.6f}" for n in names]
            )
    write_report(results, manifest, out_dir)
    print(f"ladder -> {out_dir}/report.md ({time.time() - t0:.0f}s)")
    return 0


class CodeEmbeddingScorer:
    """Cosine over docstring-stripped source embeddings, memoized per skill id."""

    def __init__(self, embedder: Any) -> None:
        self._e = embedder
        self._cache: dict[str, np.ndarray] = {}

    def _vec(self, unit: Unit) -> np.ndarray | None:
        if unit.code is None:
            return None
        if unit.id not in self._cache:
            self._cache[unit.id] = np.asarray(
                self._e.embed(unit.code.source_no_doc), dtype=np.float32
            )
        return self._cache[unit.id]

    def __call__(self, a: Unit, b: Unit) -> float:
        va, vb = self._vec(a), self._vec(b)
        if va is None or vb is None:
            return 0.0
        sim = float(np.clip(np.dot(va, vb), -1.0, 1.0))
        return (sim + 1.0) / 2.0


def categorical_summary(
    pred: list[bool],
    truth: list[bool],
    hard_neg: list[bool],
    *,
    abstain: list[bool] | None = None,
    note: str = "",
) -> dict[str, Any]:
    """A rung with no threshold to sweep (categorical / calibrated).

    Abstentions are EXCLUDED from the decision counts and reported as their own rate — folding
    them into DISTINCT would make a rung look safest exactly where it failed.
    """
    abstain = abstain or [False] * len(pred)
    tp = fp = fn = tn = fm_hard = 0
    n_abstain = 0
    for p, t, h, ab in zip(pred, truth, hard_neg, abstain, strict=True):
        if ab:
            n_abstain += 1
            continue
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
            if h:
                fm_hard += 1
        elif not p and t:
            fn += 1
        else:
            tn += 1
    n_pred = tp + fp
    n_hard = sum(hard_neg)
    lo, hi = clopper_pearson(fm_hard, n_hard)
    hi1 = clopper_pearson_upper_one_sided(fm_hard, n_hard)
    return {
        "kind": "categorical",
        "note": note,
        "point": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "fdr_merge": (fp / n_pred) if n_pred else None,
            "fpr_distinct": (fp / (fp + tn)) if (fp + tn) else None,
            "precision_non_equiv": (tn / (tn + fn)) if (tn + fn) else None,
            "recall_merge": (tp / (tp + fn)) if (tp + fn) else None,
            "fm_hard": fm_hard,
            "n_hard": n_hard,
            "fm_hard_ci95_two_sided": [round(lo, 4), round(hi, 4)],
            "fm_hard_upper95_one_sided": round(hi1, 4),
            "abstain": n_abstain,
            "abstain_rate": n_abstain / len(pred) if pred else 0.0,
        },
    }


def run_llm_source(
    pairs: list[tuple[str, str]],
    units: dict[str, Unit],
    truth: list[bool],
    hard_neg: list[bool],
    args: Any,
    out_dir: Path,
) -> dict[str, Any]:
    """Rung 7 — a frontier LLM reading both implementations, repeated to measure dispersion.

    The reply is parsed to a relation word. An unparseable or empty reply is ABSTAIN, never
    DISTINCT (A2/01 §3 rung 7a). "Deterministic decoding with three seeds" is incoherent, so what
    is measured here is repeat-call nondeterminism: the identical call is issued N times and the
    observed spread is reported as a measured property (rung 7b).
    """
    from viscurate.baselines.code_judges import LlmSourceJudge

    idx = list(range(len(pairs)))
    capped = False
    if args.llm_max_pairs and len(idx) > args.llm_max_pairs:
        # Cap the HARD slice in, then fill deterministically — and log that it happened, because a
        # silent truncation reads as full coverage.
        hard_idx = [i for i in idx if hard_neg[i]]
        pos_idx = [i for i in idx if truth[i] and i not in hard_idx]
        rest = [i for i in idx if i not in hard_idx and i not in pos_idx]
        keep = hard_idx + pos_idx
        keep += rest[: max(0, args.llm_max_pairs - len(keep))]
        idx = sorted(set(keep))
        capped = True
        print(f"  7-llm-on-source: CAPPED to {len(idx)}/{len(pairs)} pairs (hard+positives kept)")

    client = OpenAIClient(
        args.llm_source_model, base_url=args.llm_base_url, max_tokens=2000, timeout=180
    )
    # repeats=1: the judge's own majority vote would HIDE the dispersion this rung must report,
    # so one call per rep here and the spread is computed across reps below.
    judge = LlmSourceJudge(client=client, repeats=1)
    reps: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for rep in range(args.llm_repeats):
        pred: list[bool] = []
        abstain: list[bool] = []
        t_sub = [truth[i] for i in idx]
        h_sub = [hard_neg[i] for i in idx]
        for i in idx:
            a, b = pairs[i]
            ua, ub = units[a], units[b]
            if ua.code is None or ub.code is None:
                pred.append(False)
                abstain.append(True)
                continue
            try:
                v: JudgeVerdict = judge.verdict(ua.code, ub.code)
                is_abstain = v.relation is Relation.UNCERTAIN
                pred.append(bool(v.mergeable) and not is_abstain)
                abstain.append(is_abstain)
            except Exception as exc:
                pred.append(False)
                abstain.append(True)
                raw.append(
                    {"rep": rep, "pair": [a, b], "error": f"{type(exc).__name__}: {exc}"[:200]}
                )
        summary = categorical_summary(pred, t_sub, h_sub, abstain=abstain, note=f"rep {rep}")
        reps.append(summary["point"])
        print(f"    rep {rep}: {json.dumps(summary['point'])}")
    (out_dir / "llm_source_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    def spread(key: str) -> dict[str, Any]:
        vals = [r[key] for r in reps if r[key] is not None]
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": statistics.mean(vals),
            "min": min(vals),
            "max": max(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    return {
        "kind": "categorical_repeated",
        "note": f"model={args.llm_source_model}, {args.llm_repeats} identical repeats",
        "capped": capped,
        "n_pairs_used": len(idx),
        "n_pairs_total": len(pairs),
        "reps": reps,
        "dispersion": {k: spread(k) for k in ("fdr_merge", "fpr_distinct", "fm_hard", "abstain")},
    }


def _library_versions() -> dict[str, str]:
    """Versions of everything a rung's score depends on. Missing optional deps record as absent
    rather than being omitted, so a rung reported as "not run" can be told apart from one whose
    dependency was silently a different version."""
    import numpy

    out: dict[str, str] = {"numpy": numpy.__version__}
    for mod in ("torch", "transformers", "sentence_transformers", "sklearn", "scipy"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except ImportError:
            out[mod] = "not-installed"
    return out


def write_report(results: dict[str, Any], manifest: dict[str, Any], out_dir: Path) -> None:
    L: list[str] = ["# A2 — the baseline ladder at matched operating points", ""]
    L += [
        f"- pair set: **{manifest['n_pairs']}** pairs (`{manifest['pairs_csv']}`)",
        f"- positive class: **{manifest['positive_class']}**",
        f"- mergeable positives present in truth: **{manifest['n_mergeable_positives']}**",
        f"- hard negatives: **{manifest['n_hard_negatives']}**",
        f"- matched rule: {manifest['matched_rule']}",
        "",
    ]
    if manifest["n_mergeable_positives"] == 0:
        L += [
            "> ⚠ **This pair set has NO mergeable positives.** Recall, precision, F1 and AUPRC for "
            "the positive class are therefore *undefined*, not zero — the shipped report prints "
            "`0.000/0.000/0.000` for them, which reads as a measured failure rather than an empty "
            "class. Only the safety columns (FDR on merges, false merges on distinct pairs, false "
            "merges on hard negatives) are interpretable here. Merge recall is estimable only on "
            "`G_rho`.",
            "",
        ]
    L += [
        "## Matched-operating-point comparison",
        "",
        "| rung | op. point | FDR_merge | FPR_distinct | precision_non_equiv | FM_hard [95% CI] "
        "| abstain | AUPRC |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(x: Any, nd: int = 4) -> str:
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    for name in sorted(results):
        r = results[name]
        if "not_run" in r:
            L.append(f"| {name} | **not run** | — | — | — | — | — | — |")
            continue
        if r["kind"] == "threshold":
            # When the matched rule is inapplicable (no mergeable positives in this pair set) the
            # shipped-default point is the only interpretable one, so report that and say so.
            m = (
                r["default"]
                if r["matched"].get("matched_status") == "inapplicable_no_positive_support"
                else r["matched"]
            )
            lo, hi = clopper_pearson(m["fm_hard"], m["n_hard"])
            L.append(
                f"| {name} | τ={m['tau']:.4f} ({m['matched_status']}) | {fmt(m['fdr_merge'])} | "
                f"{fmt(m['fpr_distinct'])} | {fmt(m['precision_non_equiv'])} | "
                f"{m['fm_hard']}/{m['n_hard']} [{lo:.3f}, {hi:.3f}] "
                f"(1-sided ≤{clopper_pearson_upper_one_sided(m['fm_hard'], m['n_hard']):.3f}) "
                f"| 0 | {fmt(r['auprc'])} |"
            )
        elif r["kind"] == "categorical":
            p = r["point"]
            L.append(
                f"| {name} | categorical | {fmt(p['fdr_merge'])} | {fmt(p['fpr_distinct'])} | "
                f"{fmt(p['precision_non_equiv'])} | {p['fm_hard']}/{p['n_hard']} "
                f"[{p['fm_hard_ci95_two_sided'][0]:.3f}, {p['fm_hard_ci95_two_sided'][1]:.3f}] "
                f"(1-sided ≤{p['fm_hard_upper95_one_sided']:.3f}) | "
                f"{p['abstain']} ({p['abstain_rate']:.3f}) | — |"
            )
        else:
            d = r["dispersion"]
            L.append(
                f"| {name} | {r['n_pairs_used']}/{r['n_pairs_total']} pairs × "
                f"{len(r['reps'])} repeats | mean {fmt(d['fdr_merge'].get('mean'))} | — | — | "
                f"mean {fmt(d['fm_hard'].get('mean'), 2)} "
                f"| mean {fmt(d['abstain'].get('mean'), 2)} | — |"
            )
    L.append("")
    L += [
        "## Reading this table",
        "",
        "- **FDR_merge** is the number that governs silent merges: of every pair a rung would "
        "merge, the fraction that is not actually mergeable. `—` means the rung merged nothing at "
        "its matched threshold, so the rate has no denominator.",
        "- **precision_non_equiv** is the paper's current *calibration* target. It is kept only "
        "for continuity: with this class balance, predicting DISTINCT everywhere already scores "
        f"{1 - manifest['n_mergeable_positives'] / max(manifest['n_pairs'], 1):.3f}.",
        "- **FM_hard** carries both exact Clopper-Pearson bounds: the two-sided 95% interval and "
        'the one-sided 95% upper bound, which is the form a "the rate is at most X" claim needs. '
        "On a 6-pair hard-negative slice even a perfect 0/6 admits ≤0.393 one-sided (≤0.459 as the "
        "two-sided upper end), so no sentence may assert a low false-merge rate from it without a "
        "bound attached. n≥59 is what a credible ≤5% claim requires.",
        "- rung **5a vs 5b** is the fairness check that matters most for the source-code claim: "
        "`structure-only` canonicalizes literals and is therefore blind to parameter-default drift "
        "by construction. `semantics-preserving` is the baseline the paper's claim must beat.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""T2 — the exhaustive differential oracle: a mechanical `true_relation` column.

WHY THIS EXISTS. Corpus A has no answer key: its `relation` column is the verifier's own output,
so scoring any baseline against it would be circular, and scoring against the published `direct`
rows means n=17 and treating a category-level claim as ground truth. Human adjudication is the
other option and there are no annotators. The oracle replaces them with a mechanical label that is
*strictly more thorough* than the thing it scores.

★ LEGITIMACY, STATED PRECISELY -- THIS IS NOT ONE CLAIM BUT TWO.

  * Against the TEXT and CODE baselines (ladder rungs 1-7): the oracle is **independent ground
    truth**. Those rungs never execute anything; an execution-derived label is a different
    modality entirely, so there is no shared failure mode.
  * Against the VERIFIER (rung 8): the oracle is **approximation-vs-exhaustive**, NOT independent.
    Both execute the same skills through the same adapters. The oracle's claim is only that it is
    more thorough -- denser battery, denser grid, no early exit, no cascade. That is the standard
    differential-testing argument and it must be LABELLED as such wherever the number appears,
    never passed off as an independent check.

TIERING. Tier 1 is a cheap pixel-space bound over the dense grid; only genuinely undecided pairs
reach the expensive Tier 2 backends. A pair that neither tier can settle is `AMBIGUOUS` and is
EXCLUDED from scoring with its rate reported -- that rate is itself a measurement of how much
real-library equivalence is undecidable, not a failure to report.

★ THE ORACLE IS NEVER TUNED TOWARD THE VERIFIER. If oracle and verifier disagree often, that is
the result. Adjusting eps/delta to raise agreement would invert the experiment: the oracle would
stop being an independent measurement and start being a restatement of what it is meant to score.
The thresholds below are set from the *representation* (float32 round-trip error, 8-bit
quantisation) and not from any observed agreement rate.

DONE TEST (both must pass or the oracle is broken, and the script exits non-zero):
  1. every self-pair certifies EQUIVALENT
  2. the engineered hard negatives certify DISTINCT
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# eps: 1 LSB of 8-bit quantisation (1/255) with headroom for float round-trip through adapters.
# A difference below this cannot be represented in any 8-bit image format, so calling it a
# behavioural difference would be measuring the container, not the skill.
EPS_ORACLE = 1.5 / 255.0
# delta: a difference this large is visible at a glance; nothing near a threshold lands here.
DELTA_ORACLE = 0.10

EQUIVALENT = "EQUIVALENT"
DISTINCT = "DISTINCT"
AMBIGUOUS = "AMBIGUOUS"


def linf(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())


def tier1(
    outs_a: dict[str, np.ndarray], outs_b: dict[str, np.ndarray]
) -> tuple[str, float, str, int]:
    """Worst-case L-inf over the shared probes. Returns (verdict, worst, probe, n_common)."""
    common = [p for p in outs_a if p in outs_b]
    if not common:
        # No shared successfully-executed probe = no evidence. This is exactly the hole that made
        # the verifier certify PERCEPTUAL on an empty common set; the oracle must not repeat it.
        return AMBIGUOUS, float("inf"), "", 0
    worst, wp = -1.0, ""
    for p in common:
        d = linf(outs_a[p], outs_b[p])
        if d > worst:
            worst, wp = d, p
    if worst <= EPS_ORACLE:
        return EQUIVALENT, worst, wp, len(common)
    if worst >= DELTA_ORACLE:
        return DISTINCT, worst, wp, len(common)
    return AMBIGUOUS, worst, wp, len(common)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes-dir", default="data/probe_images_full")
    ap.add_argument("--ground-truth", default="configs/ground_truth_g0.yaml")
    ap.add_argument("--alignment", default="", help="corpus-A style YAML; empty = builtin G_0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid-points", type=int, default=5, help="dense grid points per parameter")
    ap.add_argument("--max-cache-gb", type=float, default=12.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--within-family-only",
        action="store_true",
        help="score only pairs sharing a family -- the set the A2 ladder actually scores. "
        "All-pairs on Corpus A is 93C2=4278 comparisons for 182 pairs of interest.",
    )
    ap.add_argument("--out", default="results/wacv_r2/oracle")
    args = ap.parse_args()

    from viscurate.corruption.apply import load_g0_spec
    from viscurate.equivalence.backends import DinoBackend, LpipsBackend
    from viscurate.equivalence.candidates import ENGINEERED_HARD_NEGATIVES
    from viscurate.equivalence.compare import BatteryEvaluator
    from viscurate.probes.build import load_probe
    from viscurate.probes.manifest import ProbeManifest
    from viscurate.skills.library import build_builtin_registry

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = ProbeManifest.model_validate_json(
        (Path(args.probes_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    battery = [(e.probe_id, load_probe(args.probes_dir, e.probe_id)) for e in manifest.entries]

    if args.alignment:
        import yaml
        from run_corpus_a import build_skills

        alignment = yaml.safe_load(Path(args.alignment).read_text(encoding="utf-8"))
        skill_map, info = build_skills(alignment)
        skills = list(skill_map.values())
        corpus = Path(args.alignment).stem
    else:
        skills = list(build_builtin_registry().all())
        info = {}
        corpus = "g0"

    prov = BatteryEvaluator(
        skills, battery, seed=args.seed, max_cache_bytes=int(args.max_cache_gb * 1e9)
    )
    by_id = {s.id: s for s in skills}

    def outs(sid: str) -> dict[str, np.ndarray]:
        o = prov.outputs(sid, seed=args.seed)
        return {p: o.canon[p].rgb for p in o.canon}

    # ---- pair set: all within-corpus pairs, sharded ------------------------------------------
    ids = sorted(by_id)
    all_pairs = list(itertools.combinations(ids, 2))
    if args.within_family_only:
        fam = {k: v.get("family") for k, v in info.items()}
        all_pairs = [
            (a, b) for a, b in all_pairs if fam.get(a) is not None and fam.get(a) == fam.get(b)
        ]
    self_pairs = [(i, i) for i in ids]
    pairs = [p for k, p in enumerate(all_pairs) if k % args.shards == args.shard]
    print(
        f"oracle shard {args.shard}/{args.shards}: {len(pairs)} pairs of {len(all_pairs)} "
        f"over {len(ids)} skills (corpus={corpus})",
        flush=True,
    )

    perceptual = LpipsBackend(device=args.device)
    semantic = DinoBackend(device=args.device)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    cache: dict[str, dict[str, np.ndarray]] = {}

    def get(sid: str) -> dict[str, np.ndarray]:
        if sid not in cache:
            if len(cache) > 24:  # bounded; pure compute optimisation, cannot change a verdict
                cache.pop(next(iter(cache)))
            cache[sid] = outs(sid)
        return cache[sid]

    def judge(a: str, b: str) -> dict[str, Any]:
        try:
            oa, ob = get(a), get(b)
        except Exception as exc:  # an unexecutable skill is evidence-free, not equivalent
            return {"a": a, "b": b, "verdict": AMBIGUOUS, "reason": f"exec: {type(exc).__name__}"}
        v, worst, probe, ncom = tier1(oa, ob)
        row = {
            "a": a,
            "b": b,
            "verdict": v,
            "tier": 1,
            "worst_linf": worst,
            "worst_probe": probe,
            "n_common": ncom,
        }
        if v != AMBIGUOUS:
            return row
        # ---- Tier 2: only the genuinely undecided reach the learned backends -----------------
        common = [p for p in oa if p in ob]
        if not common:
            row["reason"] = "no common executed probe"
            return row
        lp = max(float(perceptual.distance(oa[p], ob[p])) for p in common)
        fa = semantic.features([oa[p] for p in common])
        fb = semantic.features([ob[p] for p in common])
        num = float((fa * fb).sum(axis=1).min())
        den = float((np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1)).min())
        dino_worst = 1.0 - (num / den if den else 0.0)
        row.update({"tier": 2, "lpips_worst": lp, "dino_worst": dino_worst})
        # Deliberately conservative: Tier 2 can only CONFIRM equivalence, never manufacture it.
        if lp <= 0.02 and dino_worst <= 0.02:
            row["verdict"] = EQUIVALENT
        elif lp >= 0.20 or dino_worst >= 0.20:
            row["verdict"] = DISTINCT
        else:
            row["verdict"] = AMBIGUOUS
        return row

    # ---- done test (shard 0 only) -------------------------------------------------------------
    gate: dict[str, Any] = {}
    if args.shard == 0:
        sp = [judge(a, b) for a, b in self_pairs]
        n_ok = sum(1 for r in sp if r["verdict"] == EQUIVALENT)
        g0_spec = load_g0_spec(args.ground_truth)
        hn = list(getattr(g0_spec, "distinct_hard_negatives", [])) or list(
            ENGINEERED_HARD_NEGATIVES
        )
        hn = [(a, b) for a, b in hn if a in by_id and b in by_id]
        hr = [judge(a, b) for a, b in hn]
        n_hn = sum(1 for r in hr if r["verdict"] == DISTINCT)
        gate = {
            "self_pairs": len(sp),
            "self_equivalent": n_ok,
            "self_pass": n_ok == len(sp),
            "hard_negatives": len(hr),
            "hard_negatives_distinct": n_hn,
            "hard_neg_pass": n_hn == len(hr),
            "failures": [
                r for r in sp + hr if r["verdict"] != (EQUIVALENT if r["a"] == r["b"] else DISTINCT)
            ][:20],
        }
        print(
            f"DONE TEST: self {n_ok}/{len(sp)} EQUIVALENT | hard negatives "
            f"{n_hn}/{len(hr)} DISTINCT",
            flush=True,
        )

    for i, (a, b) in enumerate(pairs, 1):
        rows.append(judge(a, b))
        if i % 50 == 0 or i == len(pairs):
            el = time.time() - t0
            print(f"  [{i}/{len(pairs)}] {el:.0f}s ({el / i:.2f}s/pair)", flush=True)

    from collections import Counter

    vc = Counter(r["verdict"] for r in rows)
    payload = {
        "artifact": "differential_oracle",
        "corpus": corpus,
        "shard": args.shard,
        "shards": args.shards,
        "eps_oracle": EPS_ORACLE,
        "delta_oracle": DELTA_ORACLE,
        "legitimacy": {
            "vs_text_and_code_baselines": "independent ground truth (different modality)",
            "vs_verifier": "approximation-vs-exhaustive, NOT independent -- label it as such",
        },
        "gate": gate,
        "verdicts": dict(vc),
        "ambiguous_rate": vc[AMBIGUOUS] / len(rows) if rows else None,
        "rows": rows,
    }
    dest = out_dir / f"oracle_{corpus}_shard{args.shard:03d}.json"
    dest.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {dest} | verdicts {dict(vc)}")
    if gate and not (gate["self_pass"] and gate["hard_neg_pass"]):
        print("*** ORACLE DONE TEST FAILED -- do not use these labels ***")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

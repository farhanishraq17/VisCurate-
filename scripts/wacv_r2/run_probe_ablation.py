#!/usr/bin/env python
"""T5 — probe-battery ablation: how much does the verdict depend on the probes?

Answers R3's minor weakness verbatim: *"There is no analysis of the probe battery. The method
depends heavily on the chosen probes, yet there is no ablation showing how performance changes
with the number or diversity of probes."*

WHAT IS MEASURED, on the `G_0` designed pair set (the only pairs with an authored relation):
  * agreement  -- fraction of pairs whose relation is unchanged from the full 177-probe battery
  * merge-safety -- false merges among the engineered hard negatives, with an exact interval
  * abstention -- how often the verifier declines as evidence is removed

★ THE FULL BATTERY IS THE REFERENCE, NOT AN ORACLE. "Agreement" here means *agreement with the
shipped configuration*, not correctness. A small battery that disagrees is not thereby wrong -- it
is less evidenced. The designed relations are the only correctness signal and are scored separately.

★ SELECTION LEAKAGE. A strategy that picks probes using the pairs it is then evaluated on would
leak. `stratified` and `first` are data-independent (they use only the probe class and index) so
they cannot leak. `random` is seeded and also data-independent. No strategy here inspects a
verdict, so the disjoint calibration/evaluation split the plan requires is satisfied trivially --
and that is stated rather than assumed, because a learned selector WOULD need the split.

★ SUBSETS ARE NESTED within a strategy (the |B|=8 set is a subset of |B|=16, ...). Non-nested
subsets confound "fewer probes" with "different probes"; nesting makes the curve monotone in
evidence and attributes any change to battery SIZE alone.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

SIZES = [8, 16, 32, 64, 128, 177]
MERGEABLE = {"EXACT", "PERCEPTUAL"}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes-dir", default="data/probe_images_full")
    ap.add_argument("--ground-truth", default="configs/ground_truth_g0.yaml")
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--param-alignment", default="configs/param_alignment.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-cache-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", default="results/wacv_r2/ablation")
    args = ap.parse_args()

    import yaml

    from viscurate.config import ThresholdConfig
    from viscurate.corruption.apply import load_g0_spec
    from viscurate.equivalence.backends import ClipBackend, DinoBackend, LpipsBackend
    from viscurate.equivalence.compare import BatteryEvaluator
    from viscurate.equivalence.param_alignment import load_param_alignment
    from viscurate.equivalence.taxonomy import classify
    from viscurate.probes.build import load_probe
    from viscurate.probes.manifest import ProbeManifest
    from viscurate.skills.library import build_builtin_registry

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = ProbeManifest.model_validate_json(
        (Path(args.probes_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    entries = list(manifest.entries)
    full = [(e.probe_id, load_probe(args.probes_dir, e.probe_id)) for e in entries]
    cls_of = {e.probe_id: str(e.probe_id).rsplit("_", 1)[0] for e in entries}

    thr_doc = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8"))
    thr = ThresholdConfig.model_validate(thr_doc.get("thresholds", thr_doc))
    if not thr.calibrated:
        raise SystemExit(f"{args.thresholds} is not calibrated; refusing to score against it")

    g0 = load_g0_spec(args.ground_truth)
    skills = list(build_builtin_registry().all())
    by_id = {s.id: s for s in skills}

    def norm(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    truth: dict[tuple[str, str], str] = {}
    for a, b in g0.semantic_preserving:
        truth[norm(a, b)] = "SEMANTIC_PRESERVING"
    for a, b in g0.complementary:
        truth[norm(a, b)] = "COMPLEMENTARY"
    for d in g0.subsumption:
        truth[norm(d.spec, d.gen)] = "SUBSUMPTION"
    hard = {norm(a, b) for a, b in g0.distinct_hard_negatives}
    for p in hard:
        truth[p] = "DISTINCT"
    pairs = [p for p in sorted(truth) if p[0] in by_id and p[1] in by_id]
    print(f"designed pairs: {len(pairs)} ({len(hard)} hard negatives)", flush=True)

    # ---- nested subsets per strategy ----------------------------------------------------------
    ids = [e.probe_id for e in entries]

    def subset(strategy: str, n: int) -> list[str]:
        if strategy == "first":
            return ids[:n]
        if strategy == "random":
            r = random.Random(args.seed)
            order = list(ids)
            r.shuffle(order)
            return order[:n]
        # stratified: round-robin over probe classes, so every class is represented at every size
        buckets: dict[str, list[str]] = {}
        for i in ids:
            buckets.setdefault(cls_of[i], []).append(i)
        keys, out, k = sorted(buckets), [], 0
        while len(out) < n and any(buckets[x] for x in keys):
            b = buckets[keys[k % len(keys)]]
            if b:
                out.append(b.pop(0))
            k += 1
        return out

    perceptual = LpipsBackend(device=args.device)
    semantic = DinoBackend(device=args.device)
    clip_b = ClipBackend(device=args.device)
    alignment = load_param_alignment(args.param_alignment)

    BACKENDS = {
        "full": dict(perceptual=perceptual, semantic=semantic, clip=clip_b),
        "lpips_only": dict(perceptual=perceptual, semantic=None, clip=None),
        "dino_only": dict(perceptual=None, semantic=semantic, clip=None),
        "clip_only": dict(perceptual=None, semantic=None, clip=clip_b),
        "pixel_only": dict(perceptual=None, semantic=None, clip=None),
    }

    configs: list[dict[str, Any]] = []
    for strat in ("stratified", "first", "random"):
        for n in SIZES:
            configs.append({"kind": "battery", "strategy": strat, "n_probes": n, "backend": "full"})
    for bk in BACKENDS:
        if bk != "full":
            configs.append(
                {"kind": "backend", "strategy": "stratified", "n_probes": 177, "backend": bk}
            )
    configs = [c for i, c in enumerate(configs) if i % args.shards == args.shard]
    print(f"shard {args.shard}/{args.shards}: {len(configs)} configs", flush=True)

    results: list[dict[str, Any]] = []
    t0 = time.time()
    for cfg in configs:
        sel = set(subset(cfg["strategy"], cfg["n_probes"]))
        bat = [(pid, im) for pid, im in full if pid in sel]
        prov = BatteryEvaluator(
            skills, bat, seed=args.seed, max_cache_bytes=int(args.max_cache_gb * 1e9)
        )
        bk = BACKENDS[cfg["backend"]]
        verdicts: dict[tuple[str, str], str] = {}
        errs = 0
        for a, b in pairs:
            try:
                r = classify(
                    by_id[a].comparator_view(),
                    by_id[b].comparator_view(),
                    prov,
                    thresholds=thr,
                    alignment=alignment,
                    seed=args.seed,
                    **bk,
                )
                verdicts[(a, b)] = str(r.relation)
            except Exception:
                errs += 1
        fm = [p for p in hard if verdicts.get(p) in MERGEABLE]
        n_hard = len([p for p in hard if p in verdicts])
        correct = sum(1 for p, v in verdicts.items() if v == truth[p])
        row = {
            **cfg,
            "n_scored": len(verdicts),
            "n_errors": errs,
            "designed_correct": correct,
            "designed_accuracy": correct / len(verdicts) if verdicts else None,
            "hard_negatives_scored": n_hard,
            "false_merges_hard": len(fm),
            "false_merges_ci95": clopper_pearson(len(fm), n_hard) if n_hard else None,
            "n_uncertain": sum(1 for v in verdicts.values() if v == "UNCERTAIN"),
            "relation_counts": dict(Counter(verdicts.values())),
            "verdicts": {f"{a}|{b}": v for (a, b), v in verdicts.items()},
            "wall_s": round(time.time() - t0, 1),
        }
        results.append(row)
        print(
            f"  [{cfg['kind']}] {cfg['strategy']}/{cfg['n_probes']}/{cfg['backend']}: "
            f"designed {correct}/{len(verdicts)} · false merges {len(fm)}/{n_hard} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        del prov

    dest = out_dir / f"ablation_shard{args.shard:02d}.json"
    dest.write_text(
        json.dumps(
            {
                "artifact": "probe_battery_ablation",
                "shard": args.shard,
                "shards": args.shards,
                "sizes": SIZES,
                "nested_subsets": True,
                "reference": "full battery (177, stratified) -- a REFERENCE, not an oracle",
                "leakage": "all strategies are data-independent (class/index/seed only); no "
                "strategy inspects a verdict, so calibration and evaluation cannot overlap",
                "recalibrated": False,
                "results": results,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

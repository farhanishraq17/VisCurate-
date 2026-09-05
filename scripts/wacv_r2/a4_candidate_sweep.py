#!/usr/bin/env python
"""A4.1 + A4.5 — candidate screening: reduction AND recall vs fingerprint radius, and |C| scaling.

WHY THIS RUN EXISTS. The A4 deliverable asks for reduction *and recall* as a function of the
fingerprint radius. The recorded telemetry contains exactly ONE radius (0.5) plus a `-1.0`
sentinel, so the curve has never existed. This is not a telemetry backfill of an old run — it is
the A4.1 experiment, which was never performed. Candidate generation is fingerprint-distance only
(no verification), so the whole sweep is cheap.

★ RECALL NEEDS REAL POSITIVES, AND `G_0` HAS NONE. `G_0` designates zero EXACT and zero PERCEPTUAL
pairs, so "recall of mergeable pairs" is undefined there and any number reported for it would be
vacuous. `G_rho` injects duplicates and therefore has genuine mergeable positives, so the honest
recall curve is measured on `G_rho`. `G_0` still supplies the reduction curve and a *designed-pair*
retention curve, which is labelled as such and never called merge recall.

The answer key is repaired exactly as in `run_grho_replay.py` (materialise EXACT-clone inheritance,
exclude PERCEPTUAL-clone inheritance) so the positives counted here are the same population the
verifier was scored against.

A4.5: |C| growth is fitted over library sizes by SUBSAMPLING `G_0` (nested subsets, recorded seed)
rather than asserting a complexity class. Nested subsets keep the curve monotone and are the
honest way to vary n while holding the skill distribution fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_grho_replay import MERGEABLE, repair_key

from viscurate.corruption.apply import apply_corruption, load_g0_spec
from viscurate.corruption.plan import CorruptionLog
from viscurate.equivalence.backends import DinoBackend
from viscurate.equivalence.candidates import candidate_pairs, compute_fingerprints
from viscurate.equivalence.compare import BatteryEvaluator
from viscurate.probes.build import load_probe
from viscurate.probes.manifest import ProbeManifest
from viscurate.skills.library import build_builtin_registry

RADII = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]
KS = [1, 3, 5, 10, 20, 50]


def norm(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def sweep(
    views: list[Any],
    fps: dict[str, Any],
    positives: set[tuple[str, str]],
    labelled: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Reduction + recall at every (k, radius). `positives` may be empty -- then recall is None."""
    n = len(views)
    n_all = n * (n - 1) // 2
    out = []
    for k in KS:
        for r in RADII:
            t0 = time.perf_counter()
            # hard_negatives=() so the curve measures the FINGERPRINT screen alone; the engineered
            # injection is a separate, always-on union and would otherwise floor the recall.
            cand = candidate_pairs(
                views, fps, k=k, max_distance=r, include_same_family=False, hard_negatives=()
            )
            cand = {norm(a, b) for a, b in cand}
            row = {
                "k": k,
                "radius": r,
                "n_skills": n,
                "n_all_pairs": n_all,
                "n_candidates": len(cand),
                "reduction": 1.0 - (len(cand) / n_all) if n_all else 0.0,
                "wall_ms": (time.perf_counter() - t0) * 1000.0,
            }
            if positives:
                hit = len(cand & positives)
                row["n_positives"] = len(positives)
                row["positives_retained"] = hit
                row["merge_recall"] = hit / len(positives)
            if labelled:
                row["n_labelled"] = len(labelled)
                row["labelled_retained"] = len(cand & labelled)
                row["labelled_retention"] = len(cand & labelled) / len(labelled)
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes-dir", default="data/probe_images_full")
    ap.add_argument("--ground-truth", default="configs/ground_truth_g0.yaml")
    ap.add_argument("--corruption-dir", default="data/corruption")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--screening", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-cache-gb", type=float, default=3.0)
    ap.add_argument(
        "--grho-instances",
        default="",
        help="comma-separated instance names; default = one per rho level (seed1234, mixed)",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--skip-g0", action="store_true")
    ap.add_argument("--out", default="results/wacv_r2/a4_sweep")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    g0_spec = load_g0_spec(args.ground_truth)
    l0 = build_builtin_registry().all()
    manifest = ProbeManifest.model_validate_json(
        (Path(args.probes_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    battery = [(e.probe_id, load_probe(args.probes_dir, e.probe_id)) for e in manifest.entries]
    probe_ids = [p for p, _ in battery]
    semantic = DinoBackend(device=args.device)

    result: dict[str, Any] = {"artifact": "a4_candidate_sweep", "shard": args.shard}

    # ---- G_0: reduction curve + designed-pair retention + |C| scaling ------------------------
    if args.shard == 0 and not args.skip_g0:
        views = [s.comparator_view() for s in l0]
        prov = BatteryEvaluator(
            list(l0), battery, seed=args.seed, max_cache_bytes=int(args.max_cache_gb * 1e9)
        )
        t0 = time.time()
        fps = compute_fingerprints(
            views, prov, semantic, screening_ids=probe_ids[: args.screening], seed=args.seed
        )
        print(f"g0 fingerprints: {len(fps)} in {time.time() - t0:.1f}s", flush=True)

        designed: set[tuple[str, str]] = set()
        for a, b in g0_spec.semantic_preserving:
            designed.add(norm(a, b))
        for a, b in g0_spec.complementary:
            designed.add(norm(a, b))
        for a, b in g0_spec.distinct_hard_negatives:
            designed.add(norm(a, b))
        for d in g0_spec.subsumption:
            designed.add(norm(d.spec, d.gen))
        for a, b in g0_spec.exact:
            designed.add(norm(a, b))
        for a, b in g0_spec.perceptual:
            designed.add(norm(a, b))

        g0_pos = {norm(a, b) for a, b in list(g0_spec.exact) + list(g0_spec.perceptual)}
        result["g0"] = {
            "n_skills": len(views),
            "n_designed": len(designed),
            "n_mergeable_positives": len(g0_pos),
            "sweep": sweep(views, fps, g0_pos, designed),
        }

        # A4.5 -- |C| growth over nested subsamples at the shipped operating point
        rng = random.Random(args.seed)
        order = [v.id for v in views]
        rng.shuffle(order)
        by_id = {v.id: v for v in views}
        scaling = []
        for n in [10, 20, 30, 40, 50, 60, 80, 100]:
            if n > len(order):
                continue
            sub = [by_id[i] for i in order[:n]]
            subfp = {i: fps[i] for i in order[:n]}
            c = candidate_pairs(
                sub, subfp, k=5, max_distance=0.5, include_same_family=False, hard_negatives=()
            )
            scaling.append(
                {
                    "n_skills": n,
                    "n_all_pairs": n * (n - 1) // 2,
                    "n_candidates": len(c),
                }
            )
            print(f"  scaling n={n}: |C|={len(c)}", flush=True)
        result["g0_scaling"] = {
            "note": "nested random subsamples of G_0, seed recorded; k=5, radius=0.5",
            "seed": args.seed,
            "points": scaling,
        }
        # log-log fit |C| ~ a * n^b over the measured points
        xs = [math.log(p["n_skills"]) for p in scaling if p["n_candidates"] > 0]
        ys = [math.log(p["n_candidates"]) for p in scaling if p["n_candidates"] > 0]
        if len(xs) >= 3:
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
            sxx = sum((x - mx) ** 2 for x in xs)
            b = sxy / sxx if sxx else float("nan")
            a = math.exp(my - b * mx)
            ss_res = sum((y - (math.log(a) + b * x)) ** 2 for x, y in zip(xs, ys, strict=True))
            ss_tot = sum((y - my) ** 2 for y in ys)
            result["g0_scaling"]["fit"] = {
                "form": "|C| = a * n^b",
                "a": a,
                "b": b,
                "r2": 1 - ss_res / ss_tot if ss_tot else None,
                "n_points": len(xs),
                "projected_C_at_1e4": a * (1e4**b),
                "all_pairs_at_1e4": 1e4 * (1e4 - 1) / 2,
            }
        del prov

    # ---- G_rho: the real merge-recall-vs-radius curve -----------------------------------------
    if args.grho_instances:
        names = [n.strip() for n in args.grho_instances.split(",") if n.strip()]
    else:
        names = [
            f"rho{r:03d}_uniform_seed1234_mixed" for r in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ]
    names = [n for i, n in enumerate(sorted(names)) if i % args.shards == args.shard]

    grho_rows = []
    for name in names:
        p = Path(args.corruption_dir) / name / "corruption_log.json"
        if not p.exists():
            print(f"  SKIP missing {name}", flush=True)
            continue
        log = CorruptionLog.model_validate_json(p.read_text(encoding="utf-8"))
        lib = apply_corruption(l0, log, g0_spec)
        skills = list(lib.registry.all())
        labelled, _excluded = repair_key(lib.g_rho_spec, g0_spec)
        ids = {s.id for s in skills}
        pos = {
            norm(a, b)
            for (a, b), rel in labelled.items()
            if rel in MERGEABLE and a in ids and b in ids
        }
        lab = {norm(a, b) for (a, b) in labelled if a in ids and b in ids}
        views = [s.comparator_view() for s in skills]
        prov = BatteryEvaluator(
            skills, battery, seed=args.seed, max_cache_bytes=int(args.max_cache_gb * 1e9)
        )
        t0 = time.time()
        fps = compute_fingerprints(
            views, prov, semantic, screening_ids=probe_ids[: args.screening], seed=args.seed
        )
        rows = sweep(views, fps, pos, lab)
        for r in rows:
            r["instance"] = name
            r["rho"] = int(name.split("_")[0].removeprefix("rho")) / 100
        grho_rows.extend(rows)
        print(
            f"  {name}: {len(views)} skills, {len(pos)} mergeable positives, "
            f"fp {time.time() - t0:.1f}s",
            flush=True,
        )
        del prov

    result["grho"] = grho_rows
    dest = out_dir / f"sweep_shard{args.shard:02d}.json"
    dest.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

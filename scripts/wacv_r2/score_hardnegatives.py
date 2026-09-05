#!/usr/bin/env python
"""Score ONE shard of the expanded hard-negative slice, at a carried-over operating point.

WHY A DEDICATED RUNNER. ``viscurate run-benchmark`` always scores the union of the designed pairs
and the fingerprint-NN candidate set (~1000 pairs, ~10 h on this hardware). The headline safety
claim only needs the hard-negative pairs themselves, and ``run_benchmark`` already accepts an
explicit ``pairs=`` list which bypasses candidate generation entirely, which is the real win: the
~1000-pair fingerprint-NN candidate set never gets scored.

⚠ ``--shards/--shard`` DOES NOT reduce per-process work here, and the docstring used to claim it
did. ``run_benchmark`` computes ``scored = generated | designed`` — the answer key's pairs are
always scored, by design — so passing a 12-pair shard as ``pairs=`` still scores all 87 designed
pairs of the expanded key. Every shard therefore does the same 87 pairs. Wall-clock is still one
shard's time because they run concurrently, and the redundancy is not useless: six independent
processes scoring an identical pair set is a free cross-shard DETERMINISM check, which the merge
step asserts. But it is not a 6x speedup and must not be described as one.

NOTHING IS CALIBRATED HERE. ``--thresholds`` must name an already-calibrated file and its
provenance is copied into the shard manifest.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from viscurate.benchmark.ground_truth import load_ground_truth
from viscurate.benchmark.runner import run_benchmark
from viscurate.config import ThresholdConfig
from viscurate.equivalence.backends import ClipBackend, DinoBackend, LpipsBackend
from viscurate.equivalence.compare import BatteryEvaluator
from viscurate.equivalence.param_alignment import load_param_alignment
from viscurate.probes.build import load_probe
from viscurate.probes.manifest import ProbeManifest
from viscurate.skills.library import build_builtin_registry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ground-truth", default="configs/ground_truth_g0_hardneg69.yaml")
    ap.add_argument("--param-alignment", default="configs/param_alignment.yaml")
    ap.add_argument("--probes-dir", default="data/probe_images_full")
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-cache-entries", type=int, default=48)
    ap.add_argument(
        "--max-cache-gb",
        type=float,
        default=12.0,
        help="LRU bound on RETAINED BYTES. This is the bound that matters: an entry-count bound "
        "says nothing about size, and the canvas-expanding skills in this slice (rotate_*, "
        "resize_*, scale_xy, shear_*, pad_*, tile_2x2) produce OutputSets many times the average, "
        "so 48 entries reached tens of GB and was OOM-killed at pair 62 of 87. compare.py's own "
        "docstring flags exactly this case as why the byte bound exists.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    thr_doc = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8"))
    thr_fields = thr_doc.get("thresholds", thr_doc)
    thresholds = ThresholdConfig.model_validate(thr_fields)
    if not thresholds.calibrated:
        raise SystemExit(f"{args.thresholds} is not calibrated; refusing to score against it")

    gt = load_ground_truth(args.ground_truth)
    registry = build_builtin_registry()
    skills = registry.all()
    valid = {s.id for s in skills}
    hard = sorted(
        p
        for p in gt.designed_pairs()
        if gt.is_hard_negative(*p) and p[0] in valid and p[1] in valid
    )
    # Deterministic, contiguous-stride sharding so shard membership is reproducible from the args
    # alone and the union over shards is exactly the slice.
    mine = [p for i, p in enumerate(hard) if i % args.shards == args.shard]
    print(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(hard)} hard-negative pairs")
    if not mine:
        return 0

    manifest = ProbeManifest.model_validate_json(
        (Path(args.probes_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    battery = [(e.probe_id, load_probe(args.probes_dir, e.probe_id)) for e in manifest.entries]
    provider = BatteryEvaluator(
        skills,
        battery,
        seed=1234,
        max_cache_entries=(args.max_cache_entries or None),
        max_cache_bytes=(int(args.max_cache_gb * 1_000_000_000) or None),
    )
    perceptual = LpipsBackend(device=args.device)
    semantic = DinoBackend(device=args.device)
    clip = ClipBackend(device=args.device) if args.clip else None

    t0 = time.time()
    result = run_benchmark(
        [s.to_spec() for s in skills],
        provider,
        gt,
        thresholds=thresholds,
        # Text judges are evaluated separately by the A2 ladder off the same pair list, so the
        # output-grounded track is the only one this shard needs to compute.
        text_judges=[],
        alignment=load_param_alignment(args.param_alignment),
        perceptual=perceptual,
        semantic=semantic,
        clip=clip,
        pairs=set(mine),
        seed=1234,
        progress=lambda i, n, pair: print(f"  [{i}/{n}] {pair[0]} vs {pair[1]}", flush=True),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for outcome in result.outcomes:
        a, b = outcome.pair
        rows.append(
            {
                "a": a,
                "b": b,
                "true_relation": str(outcome.truth.relation),
                "is_hard_negative": bool(outcome.truth.is_hard_negative),
                "output_relation": str(outcome.output.relation),
                "output_mergeable": bool(outcome.output.mergeable),
                "reason": outcome.output.reason,
            }
        )
    (out_dir / f"shard{args.shard:02d}.json").write_text(
        json.dumps(
            {
                "shard": args.shard,
                "shards": args.shards,
                "n_pairs": len(rows),
                "ground_truth": args.ground_truth,
                "thresholds_file": args.thresholds,
                "thresholds": thr_fields,
                "recalibrated": False,
                "battery_n": len(battery),
                "max_cache_gb": args.max_cache_gb,
                "device": args.device,
                "clip": bool(args.clip),
                "wall_seconds": round(time.time() - t0, 1),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fm = sum(1 for r in rows if r["output_mergeable"])
    print(f"shard {args.shard}: {fm} false merges of {len(rows)} in {time.time() - t0:.0f}s")
    for backend in (perceptual, semantic, clip):
        if backend is not None and hasattr(backend, "close"):
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""A2 · `G_rho` replay — validate the verifier on the two relations that authorise deletion.

WHY THIS RUN EXISTS. `G_0` is a clean library: it designates zero `EXACT` and zero `PERCEPTUAL`
pairs, so the mergeable positive class is EMPTY there and merge RECALL has never been measured for
anything. `G_rho` injects duplicates, so it is the only set with `EXACT`/`PERCEPTUAL` support. This
is R3's *"the verifier itself is not validated"* complaint: today the verifier is validated on
everything except the two relations that can destroy a capability.

★ THE ANSWER KEY IS INCOMPLETE AND IS REPAIRED HERE FIRST. `g_rho.json` records the relations each
defect *injects* but not the INCIDENTAL ones the injection creates. Cloning skill `S` also creates
every relation `S` already had with the rest of the library. Audited over all 300 instances:

    1,546 unrecorded incidental relations against 12,790 recorded labels (~12%),
    in 280 of 300 instances (93.3%)
    -> semantic_preserving 816 · subsumption 420 · hard-negative 282 · complementary 28

Unlabelled pairs default to `DISTINCT`, so running the replay against the raw key would penalise
the verifier for ~1,500 CORRECT verdicts. The repair splits by whether the inheritance is sound:

  * clone is EXACT (byte-identical outputs)  -> relations transfer exactly. MATERIALISED (868).
  * clone is PERCEPTUAL (within tolerance)   -> relations are threshold-based and could flip near
                                               a threshold, so inheritance is NOT derivable.
                                               EXCLUDED from scoring, never guessed (678).

SCORING SET. Only pairs the (repaired) key labels explicitly. Precision numbers here are therefore
*within the labelled set* — the false-merge rate over the distinct sea stays `G_0`'s 0/926 and the
expanded slice's 0/69, and is not re-estimated here. Stated rather than left implicit.

UNCERTAINTY. Bootstrap resamples CORRUPTION INSTANCES, never pairs: pairs inside an instance share
base skills and a defect draw, so pair-level resampling badly understates variance.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from viscurate.config import ThresholdConfig
from viscurate.corruption.apply import apply_corruption, load_g0_spec
from viscurate.corruption.plan import CorruptionLog
from viscurate.equivalence.compare import BatteryEvaluator
from viscurate.equivalence.param_alignment import load_param_alignment
from viscurate.equivalence.relations import Relation
from viscurate.equivalence.taxonomy import classify
from viscurate.probes.build import load_probe
from viscurate.probes.manifest import ProbeManifest
from viscurate.skills.library import build_builtin_registry

RELATIONS = [r.value for r in Relation]
MERGEABLE = {Relation.EXACT.value, Relation.PERCEPTUAL.value}


def _mem_gb() -> tuple[float, float]:
    """Current RSS and PEAK RSS (``VmHWM``), in GB.

    Logging current RSS alone was a measurement error that cost real debugging time: it is sampled
    *after* the per-instance cache clear and ``gc.collect()``, so it reports the TROUGH and can
    never show the in-instance peak. Instances that died were invisible in a log full of flat
    6.6 GB readings, which is why three successive hypotheses about *which* instances were heavy
    all failed to predict the next failure. ``VmHWM`` is the high-water mark since process start,
    which is the number a step limit is actually compared against.
    """
    cur = peak = 0.0
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    cur = int(line.split()[1]) / 1048576
                elif line.startswith("VmHWM:"):
                    peak = int(line.split()[1]) / 1048576
    except OSError:
        pass
    return cur, peak


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta

    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def repair_key(
    g_rho_spec: Any, g0_spec: Any
) -> tuple[dict[tuple[str, str], str], set[tuple[str, str]]]:
    """Return (labelled pairs -> relation, excluded pairs).

    Materialises EXACT-clone inheritance; excludes PERCEPTUAL-clone inheritance.
    """

    def key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    labelled: dict[tuple[str, str], str] = {}
    for a, b in g_rho_spec.exact:
        labelled[key(a, b)] = Relation.EXACT.value
    for a, b in g_rho_spec.perceptual:
        labelled[key(a, b)] = Relation.PERCEPTUAL.value
    for d in g_rho_spec.subsumption:
        labelled[key(d.spec, d.gen)] = Relation.SUBSUMPTION.value
    for a, b in g_rho_spec.semantic_preserving:
        labelled[key(a, b)] = Relation.SEMANTIC_PRESERVING.value
    for a, b in g_rho_spec.complementary:
        labelled[key(a, b)] = Relation.COMPLEMENTARY.value
    for a, b in g_rho_spec.distinct_hard_negatives:
        labelled[key(a, b)] = Relation.DISTINCT.value

    # clone -> (original, how the clone relates to its original)
    clone_of: dict[str, tuple[str, str]] = {}
    for a, b in g_rho_spec.exact:
        if "__dup" in a and "__dup" not in b:
            clone_of[a] = (b, "exact")
        elif "__dup" in b and "__dup" not in a:
            clone_of[b] = (a, "exact")
    for a, b in g_rho_spec.perceptual:
        if "__dup" in a and "__dup" not in b:
            clone_of[a] = (b, "perceptual")
        elif "__dup" in b and "__dup" not in a:
            clone_of[b] = (a, "perceptual")

    base: list[tuple[str, str, str]] = []
    for a, b in g0_spec.semantic_preserving:
        base.append((a, b, Relation.SEMANTIC_PRESERVING.value))
    for a, b in g0_spec.complementary:
        base.append((a, b, Relation.COMPLEMENTARY.value))
    for a, b in g0_spec.distinct_hard_negatives:
        base.append((a, b, Relation.DISTINCT.value))
    sub = [(d.spec, d.gen) for d in g0_spec.subsumption]

    excluded: set[tuple[str, str]] = set()
    for clone, (orig, ctype) in clone_of.items():
        inherited: list[tuple[tuple[str, str], str]] = []
        for x, y, rel in base:
            partner = y if orig == x else (x if orig == y else None)
            if partner is not None:
                inherited.append((key(clone, partner), rel))
        for spec, gen in sub:
            if orig == spec:
                inherited.append((key(clone, gen), Relation.SUBSUMPTION.value))
            elif orig == gen:
                inherited.append((key(clone, spec), Relation.SUBSUMPTION.value))
        for k, rel in inherited:
            if k in labelled:
                continue
            if ctype == "exact":
                labelled[k] = rel
            else:
                excluded.add(k)
    return labelled, excluded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corruption-dir", default="data/corruption")
    ap.add_argument("--ground-truth", default="configs/ground_truth_g0.yaml")
    ap.add_argument("--param-alignment", default="configs/param_alignment.yaml")
    ap.add_argument("--probes-dir", default="data/probe_images_full")
    ap.add_argument("--thresholds", required=True, help="ALREADY-calibrated; never re-fit here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--per-rho", type=int, default=4, help="instances sampled per rho level")
    ap.add_argument("--sample-seed", type=int, default=1234)
    ap.add_argument(
        "--instances-file",
        default="",
        help="explicit newline-separated instance names to score, bypassing the stratified sample. "
        "Used to re-shard only the instances not yet covered, so a wider fan-out across idle nodes "
        "does no duplicate work.",
    )
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-cache-gb", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    thr_doc = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8"))
    thr_fields = thr_doc.get("thresholds", thr_doc)
    thresholds = ThresholdConfig.model_validate(thr_fields)
    if not thresholds.calibrated:
        raise SystemExit(f"{args.thresholds} is not calibrated; refusing to score against it")

    # stratified instance sample, seed recorded
    by_rho: dict[float, list[Path]] = defaultdict(list)
    for inst in sorted(Path(args.corruption_dir).iterdir()):
        if (inst / "corruption_log.json").exists():
            by_rho[int(inst.name.split("_")[0].removeprefix("rho")) / 100].append(inst)
    if args.instances_file:
        names = [
            n.strip()
            for n in Path(args.instances_file).read_text(encoding="utf-8").split()
            if n.strip()
        ]
        sample = [Path(args.corruption_dir) / n for n in names]
    else:
        rng = random.Random(args.sample_seed)
        sample = []
        for rho in sorted(by_rho):
            pool = sorted(by_rho[rho], key=lambda p: p.name)
            sample.extend(rng.sample(pool, min(args.per_rho, len(pool))))
    sample = [
        p
        for i, p in enumerate(sorted(sample, key=lambda p: p.name))
        if i % args.shards == args.shard
    ]
    print(
        f"shard {args.shard}/{args.shards}: {len(sample)} instances "
        f"(per_rho={args.per_rho}, seed={args.sample_seed})",
        flush=True,
    )

    g0_spec = load_g0_spec(args.ground_truth)
    l0 = build_builtin_registry().all()
    manifest = ProbeManifest.model_validate_json(
        (Path(args.probes_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    battery = [(e.probe_id, load_probe(args.probes_dir, e.probe_id)) for e in manifest.entries]
    alignment = load_param_alignment(args.param_alignment)

    from viscurate.equivalence.backends import ClipBackend, DinoBackend, LpipsBackend

    perceptual = LpipsBackend(device=args.device)
    semantic = DinoBackend(device=args.device)
    clip = ClipBackend(device=args.device) if args.clip else None

    rows: list[dict[str, Any]] = []
    audit = Counter()
    t0 = time.time()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"shard{args.shard:02d}.json"

    def checkpoint(n_done: int) -> None:
        """Write results after EVERY instance, atomically.

        Writing only at the end meant an OOM on instance 4 of 5 discarded four completed
        instances: 50 shards died that way and took 97 finished instances with them. A
        tmp+replace keeps the file readable by the merger at all times.
        """
        payload = {
            "artifact": "grho_replay_shard",
            "shard": args.shard,
            "shards": args.shards,
            "n_instances": n_done,
            "n_instances_planned": len(sample),
            "complete": n_done == len(sample),
            "n_pairs": len(rows),
            "per_rho": args.per_rho,
            "sample_seed": args.sample_seed,
            "thresholds_file": args.thresholds,
            "thresholds": thr_fields,
            "recalibrated": False,
            "battery_n": len(battery),
            "device": args.device,
            "clip": bool(args.clip),
            "key_repair": {
                "materialised": "EXACT-clone inheritance (sound: identical outputs)",
                "excluded": "PERCEPTUAL-clone inheritance (threshold-dependent, not derivable)",
                **dict(audit),
            },
            "scoring_set": "explicitly-labelled pairs of the REPAIRED key only",
            "wall_seconds": round(time.time() - t0, 1),
            "rows": rows,
        }
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(dest)

    for i, inst in enumerate(sample, 1):
        log = CorruptionLog.model_validate_json(
            (inst / "corruption_log.json").read_text(encoding="utf-8")
        )
        lib = apply_corruption(l0, log, g0_spec)
        skills = {s.id: s for s in lib.registry.all()}
        labelled, excluded = repair_key(lib.g_rho_spec, g0_spec)
        audit["labelled"] += len(labelled)
        audit["excluded_perceptual_inherited"] += len(excluded)
        rho = int(inst.name.split("_")[0].removeprefix("rho")) / 100
        provider = BatteryEvaluator(
            list(skills.values()),
            battery,
            seed=1234,
            max_cache_bytes=int(args.max_cache_gb * 1_000_000_000),
        )
        n_ok = 0
        for (a, b), truth in sorted(labelled.items()):
            if a not in skills or b not in skills:
                audit["skipped_missing_skill"] += 1
                continue
            try:
                res = classify(
                    skills[a].comparator_view(),
                    skills[b].comparator_view(),
                    provider,
                    thresholds=thresholds,
                    perceptual=perceptual,
                    semantic=semantic,
                    clip=clip,
                    alignment=alignment,
                    seed=1234,
                )
                rows.append(
                    {
                        "instance": inst.name,
                        "rho": rho,
                        "composition": "_".join(inst.name.split("_")[1:-2]),
                        "seed": inst.name.split("_")[-2],
                        "mode": inst.name.split("_")[-1],
                        "a": a,
                        "b": b,
                        "truth": truth,
                        "pred": str(res.relation),
                        "truth_mergeable": truth in MERGEABLE,
                        "pred_mergeable": bool(res.licenses_merge),
                        "reason": res.reason[:160],
                    }
                )
                n_ok += 1
            except Exception as exc:
                audit["classify_error"] += 1
                rows.append(
                    {
                        "instance": inst.name,
                        "rho": rho,
                        "a": a,
                        "b": b,
                        "truth": truth,
                        "pred": "ERROR",
                        "reason": f"{type(exc).__name__}: {exc}"[:160],
                    }
                )
        # A NEW evaluator is built per instance, each holding up to --max-cache-gb of cached
        # OutputSets. `del` only drops the name: without an explicit clear and a collection the
        # previous instances' caches stay reachable long enough to accumulate, and 7 instances x
        # 8 GB is exactly the 60 GB step that got OOM-killed. Clear, drop, collect — and log RSS
        # so a regression shows up as a climbing number instead of a dead job.
        provider._cache.clear()
        provider._entry_bytes.clear()
        del provider
        gc.collect()
        print(
            f"  [{i}/{len(sample)}] {inst.name} rho={rho} {n_ok} pairs "
            f"rss={_mem_gb()[0]:.1f}GB peak={_mem_gb()[1]:.1f}GB ({time.time() - t0:.0f}s)",
            flush=True,
        )
        checkpoint(i)

    for b in (perceptual, semantic, clip):
        if b is not None and hasattr(b, "close"):
            b.close()

    checkpoint(len(sample))
    ok = [r for r in rows if r["pred"] != "ERROR"]
    rec = [r for r in ok if r["truth_mergeable"]]
    hit = sum(1 for r in rec if r["pred_mergeable"])
    print(
        f"shard {args.shard}: {len(ok)} pairs; mergeable support {len(rec)}, "
        f"recall {hit}/{len(rec)} in {time.time() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

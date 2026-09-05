#!/usr/bin/env python
"""T6 / OOD — certificate-flip rate: do certified merges survive out-of-distribution inputs?

Answers the metareview's *"two skills matching on a finite probe battery may still diverge
out-of-distribution"* and R2's listing of the same point as a STRUCTURAL flaw of any output-based
approach. The objection is correct in principle; this turns it into a measured number.

★ REFUTATION DIRECTION ONLY. A pair that still agrees on `B_ood` is **not** evidence of
equivalence -- the original certificate was always conditional on the battery, and agreeing on more
inputs cannot make a finite battery infinite. Only a FLIP is informative. There is deliberately no
"reverse flip" statistic in this script: a pair that was DISTINCT on `B` and looks mergeable on
`B_ood` would be a threshold artifact, not a discovery, and reporting it would invite exactly the
misreading the reviewers are guarding against.

★ THE SUBJECTS ARE REAL CERTIFIED MERGES. The pairs re-verified here are the ones the verifier
actually licensed for deletion on real third-party code (Corpus A and the pilgram/pilgram2 fork),
not synthetic clones. A flip therefore means "we would have deleted a skill that behaves
differently on inputs the battery did not contain" -- the concrete harm the objection describes.

★ THRESHOLDS ARE CARRIED OVER UNCHANGED. Nothing is re-calibrated on `B_ood`; re-fitting on the
evaluation set is exactly the test-driven repair the plan forbids, and it would make a flip
impossible by construction.

`B_ood` classes (none of which appear in the shipped battery's distribution):
  medical    low-contrast grayscale with faint structure -- the regime where perceptual metrics
             are least calibrated
  satellite  high-frequency aerial-like texture, no dominant subject
  hdr        very wide dynamic range, most energy far from mid-grey
  aspect     extreme aspect ratios (16:1 and 1:16) -- stresses any kernel sized from the image
  tiny       4x4 and 8x8 -- kernel >= image, the highest-yield adversarial class per the plan
  saturated  fully saturated primaries and their complements
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MERGEABLE = {"EXACT", "PERCEPTUAL"}


def build_ood(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    # medical: faint low-contrast structure on a narrow grey band
    for i in range(6):
        base = np.full((128, 128), 128.0, np.float32)
        yy, xx = np.mgrid[0:128, 0:128]
        cx, cy, r = [*rng.integers(30, 98, 2).tolist(), float(rng.integers(12, 40))]
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
        img = base + 8.0 * blob + rng.normal(0, 1.2, (128, 128))
        out.append(
            (f"ood_medical_{i:03d}", np.stack([np.clip(img, 0, 255)] * 3, -1).astype(np.uint8))
        )
    # satellite: high-frequency texture, no dominant subject
    for i in range(6):
        n = rng.normal(0.5, 0.18, (128, 128, 3))
        for k in (2, 4, 8):
            n += 0.12 * np.kron(rng.normal(0.5, 0.2, (128 // k, 128 // k, 3)), np.ones((k, k, 1)))
        out.append((f"ood_satellite_{i:03d}", np.clip(n / 2.0 * 255, 0, 255).astype(np.uint8)))
    # hdr: energy pushed to both extremes
    for i in range(5):
        g = np.linspace(0, 1, 128) ** (0.25 if i % 2 else 4.0)
        img = np.stack([np.outer(g, np.ones(128))] * 3, -1)
        img = img * rng.uniform(0.9, 1.0, 3)
        out.append((f"ood_hdr_{i:03d}", np.clip(img * 255, 0, 255).astype(np.uint8)))
    # aspect: extreme ratios
    for i, (h, w) in enumerate([(8, 128), (128, 8), (4, 256), (256, 4)]):
        out.append((f"ood_aspect_{i:03d}", rng.integers(0, 256, (h, w, 3), dtype=np.uint8)))
    # tiny: kernel >= image
    for i, s in enumerate([4, 8, 4, 8]):
        out.append((f"ood_tiny_{i:03d}", rng.integers(0, 256, (s, s, 3), dtype=np.uint8)))
    # saturated primaries / complements
    for i, c in enumerate(
        [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    ):
        a = np.zeros((64, 64, 3), np.uint8)
        a[:, :] = c
        a[::4] = 255 - np.array(c, np.uint8)
        out.append((f"ood_saturated_{i:03d}", a))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes-dir", default="data/probe_images_full")
    ap.add_argument("--thresholds", required=True, help="ALREADY calibrated; never re-fit here")
    ap.add_argument("--param-alignment", default="configs/param_alignment.yaml")
    ap.add_argument(
        "--sources",
        default="scripts/wacv_r2/corpus_a_alignment.yaml,scripts/wacv_r2/corpus_a_forkpair.yaml",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--max-cache-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results/wacv_r2/ood")
    args = ap.parse_args()

    import yaml
    from run_corpus_a import build_skills

    from viscurate.config import ThresholdConfig
    from viscurate.equivalence.backends import ClipBackend, DinoBackend, LpipsBackend
    from viscurate.equivalence.compare import BatteryEvaluator
    from viscurate.equivalence.param_alignment import load_param_alignment
    from viscurate.equivalence.taxonomy import classify
    from viscurate.probes.build import load_probe
    from viscurate.probes.manifest import ProbeManifest

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest = ProbeManifest.model_validate_json(
        (Path(args.probes_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    battery_b = [(e.probe_id, load_probe(args.probes_dir, e.probe_id)) for e in manifest.entries]
    battery_ood = build_ood(rng)
    print(f"B = {len(battery_b)} probes · B_ood = {len(battery_ood)} probes", flush=True)

    thr_doc = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8"))
    thr = ThresholdConfig.model_validate(thr_doc.get("thresholds", thr_doc))
    if not thr.calibrated:
        raise SystemExit(f"{args.thresholds} is not calibrated; refusing to score against it")
    alignment = load_param_alignment(args.param_alignment)
    perceptual = LpipsBackend(device=args.device)
    semantic = DinoBackend(device=args.device)
    clip = ClipBackend(device=args.device) if args.clip else None

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for src in [s.strip() for s in args.sources.split(",") if s.strip()]:
        corpus = Path(src).stem
        align = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
        skill_map, info = build_skills(align)
        skills = list(skill_map.values())
        fam: dict[str, str] = {k: v["family"] for k, v in info.items()}
        by_fam: dict[str, list[str]] = {}
        for sid, f in fam.items():
            by_fam.setdefault(f, []).append(sid)
        pairs = [
            (a, b)
            for f, ids in by_fam.items()
            for i, a in enumerate(sorted(ids))
            for b in sorted(ids)[i + 1 :]
        ]
        prov_b = BatteryEvaluator(
            skills, battery_b, seed=args.seed, max_cache_bytes=int(args.max_cache_gb * 1e9)
        )
        prov_o = BatteryEvaluator(
            skills, battery_ood, seed=args.seed, max_cache_bytes=int(args.max_cache_gb * 1e9)
        )

        # `verdict` closes over this iteration's `skill_map` on purpose: it is only ever called
        # inside the same loop body, before `skill_map` is rebound for the next corpus.
        def verdict(prov: Any, a: str, b: str) -> tuple[str, bool, str]:
            r = classify(
                skill_map[a].comparator_view(),  # noqa: B023
                skill_map[b].comparator_view(),  # noqa: B023
                prov,
                thresholds=thr,
                perceptual=perceptual,
                semantic=semantic,
                clip=clip,
                alignment=alignment,
                seed=args.seed,
            )
            return str(r.relation), bool(r.licenses_merge), r.reason[:140]

        n_cert = 0
        for a, b in pairs:
            try:
                rel_b, merge_b, why_b = verdict(prov_b, a, b)
            except Exception as exc:
                rows.append({"corpus": corpus, "a": a, "b": b, "error_B": f"{type(exc).__name__}"})
                continue
            if not merge_b:
                continue  # only CERTIFIED merges can flip -- refutation direction only
            n_cert += 1
            try:
                rel_o, merge_o, why_o = verdict(prov_o, a, b)
            except Exception as exc:
                rows.append(
                    {
                        "corpus": corpus,
                        "a": a,
                        "b": b,
                        "relation_B": rel_b,
                        "error_OOD": f"{type(exc).__name__}: {exc}"[:160],
                        "flipped": None,
                    }
                )
                continue
            rows.append(
                {
                    "corpus": corpus,
                    "a": a,
                    "b": b,
                    "family": fam.get(a),
                    "relation_B": rel_b,
                    "relation_OOD": rel_o,
                    "licenses_merge_B": merge_b,
                    "licenses_merge_OOD": merge_o,
                    "flipped": bool(merge_b and not merge_o),
                    "reason_B": why_b,
                    "reason_OOD": why_o,
                }
            )
            print(
                f"  {corpus}: {a} vs {b}  B={rel_b} -> OOD={rel_o}"
                f"{'  ** FLIP **' if merge_b and not merge_o else ''}",
                flush=True,
            )
        print(f"  {corpus}: {n_cert} certified merges on B ({time.time() - t0:.0f}s)", flush=True)
        del prov_b, prov_o

    scored = [r for r in rows if r.get("flipped") is not None]
    flips = [r for r in scored if r["flipped"]]
    n, k = len(scored), len(flips)
    ci = (0.0, 0.0)
    if n:
        from scipy.stats import beta

        lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
        ci = (lo, hi)
        one_sided = 1.0 if k == n else float(beta.ppf(0.95, k + 1, n - k))
    else:
        one_sided = float("nan")

    payload = {
        "artifact": "ood_certificate_flip",
        "n_ood_probes": len(battery_ood),
        "ood_classes": sorted({r.split("_")[1] for r, _ in battery_ood}),
        "n_certified_on_B": n,
        "n_flipped": k,
        "certificate_flip_rate": (k / n) if n else None,
        "flip_ci95_clopper_pearson": ci,
        "flip_one_sided_95_upper": one_sided,
        "direction": "REFUTATION ONLY -- surviving B_ood is not evidence of equivalence; no "
        "reverse-flip statistic is computed.",
        "recalibrated": False,
        "rows": rows,
    }
    (out_dir / "ood_result.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(
        f"\nCERTIFICATE-FLIP: {k}/{n} = {(k / n) if n else float('nan'):.4f} "
        f"[{ci[0]:.4f}, {ci[1]:.4f}], one-sided 95% upper {one_sided:.4f}"
    )
    print(f"wrote {out_dir / 'ood_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

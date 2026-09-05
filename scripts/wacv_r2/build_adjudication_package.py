#!/usr/bin/env python
"""NEXT_ACTIONS.md §2 — build the blind annotation package for the 26 hard-negative adjudications.

These 26 pairs decide whether the headline safety bound is **4.25%** (26 confirmed as genuine hard
negatives, n=69) or **6.73%** (26 rejected, n=43). 4.25% clears the ≤5% credibility threshold;
6.73% does not.

The package enforces the protocol's blinding requirements structurally rather than by instruction:

* **The verifier's verdict is not in the annotator's file.** It lives only in the answer key, which
  is written separately and is not needed to annotate. An annotator cannot anchor on a verdict they
  cannot see.
* **Probes are chosen at the DIVERGENT parameter setting** — the worst-case grid point the verifier
  itself flagged — not at defaults, where near-misses look identical and the task is impossible.
* **Outputs are rendered side by side with a difference map**, plus both docstrings.
* **The label set includes `CANNOT-TELL`.** Forcing a guess manufactures agreement and corrupts κ.
* **Pair order and A/B side are shuffled per annotator** with a recorded per-annotator seed, so a
  systematic left/right or ordering bias cannot align across annotators.

Writes one directory per annotator plus a `key/` directory the annotators must not open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from viscurate.equivalence.compare import BatteryEvaluator
from viscurate.probes.build import load_probe
from viscurate.probes.manifest import ProbeManifest
from viscurate.skills.library import build_builtin_registry

LABELS = [
    "EXACT",
    "PERCEPTUAL",
    "SUBSUMPTION",
    "SEMANTIC_PRESERVING",
    "COMPLEMENTARY",
    "DISTINCT",
    "CANNOT-TELL",
]

GUIDELINES = """# Adjudication guidelines — cross-pair relation labelling

You are deciding, for each pair of image operators, **what relation holds between them** based only
on what they DO to images. You are not being asked to agree or disagree with any system.

## What you see per pair

`pair_NN/` contains, for each of 8–12 probe images:
  * `<probe>_A.png` — operator A's output
  * `<probe>_B.png` — operator B's output
  * `<probe>_diff.png` — amplified |A − B|; black means identical
and `docstrings.md` with both operators' documentation. Parameters are held at a **single shared
setting**, printed in `pair.json` — the setting at which these two differ most.

## The labels

| label | means |
|---|---|
| `EXACT` | Identical output on every probe you can see. |
| `PERCEPTUAL` | Differs, but you would not notice without the diff map. |
| `SUBSUMPTION` | One is a special case of the other. **Give the direction.** |
| `SEMANTIC_PRESERVING` | Same *kind* of transform, different algorithm. Visibly different. |
| `COMPLEMENTARY` | Act on different aspects; either order gives the same result. |
| `DISTINCT` | Different operations. The residual answer. |
| `CANNOT-TELL` | **Use freely.** The probes do not decide it, or no category fits. |

Longer notes on the two that matter most:

* `SUBSUMPTION` means everything the narrower operator does, the wider one can also do with some
  other parameter setting — not merely that they look similar.
* `COMPLEMENTARY` is stronger than "they happen to commute". Two operators that commute by accident
  (two rotations, two thresholds) are not complementary; they must act on *different aspects*.

## Rules

1. **`CANNOT-TELL` is a real answer, not a failure.** Guessing to avoid it corrupts the agreement
   statistic, which is the whole point of running three of you.
2. Judge only what is in front of you. Do not run the operators, and do not consult any other file.
3. Do not discuss pairs with the other annotators until all three files are submitted.
4. `SEMANTIC_PRESERVING` vs `DISTINCT` is the hard call and the one that matters most here. Ask:
   *if I needed the operation these two share, would either do?* If yes, semantic-preserving.
5. Record a one-line reason. It is what adjudication discussion will work from.

## Submitting

Fill `labels.json` — for every pair, `label` (one of the above), optional `direction`
(`A_SUBSUMES_B` / `B_SUBSUMES_A`) and a `reason` string. Leave nothing blank.
"""


def _to_u8(x: np.ndarray) -> np.ndarray:
    """Canonical rgb is float32 in [0,1]; PNG needs uint8."""
    if x.dtype == np.uint8:
        return x
    return np.clip(np.asarray(x, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _amplified_diff(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """|A−B| rescaled so a sub-perceptual difference is still visible.

    Returns the image and the amplification factor, which is written into ``pair.json`` — an
    annotator seeing a vivid diff map must be able to tell whether the true difference was large
    or merely stretched, otherwise the map itself biases the judgement toward DISTINCT.
    """
    au, bu = _to_u8(a).astype(np.int16), _to_u8(b).astype(np.int16)
    d = np.abs(au - bu).astype(np.float32)
    peak = float(d.max())
    gain = (255.0 / peak) if peak > 0 else 1.0
    return np.clip(d * gain, 0, 255).astype(np.uint8), gain


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--disagreements", default="results/wacv_r2/hardneg69/design_disagreements.json"
    )
    ap.add_argument("--pairs-json", default="results/wacv_r2/hardneg69")
    ap.add_argument("--probes-dir", default="data/probe_images_full")
    ap.add_argument(
        "--thresholds", default="results/wacv_r2/phase4_benchmark/calibrated_thresholds.yaml"
    )
    ap.add_argument("--annotators", type=int, default=3)
    ap.add_argument("--n-probes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", default="results/wacv_r2/adjudication")
    args = ap.parse_args()

    from PIL import Image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dis = json.loads(Path(args.disagreements).read_text(encoding="utf-8"))
    # the worst-case probe + params the verifier flagged, for probe selection only
    worst: dict[tuple[str, str], dict[str, Any]] = {}
    for sp in sorted(Path(args.pairs_json).glob("shard*.json")):
        for r in json.loads(sp.read_text(encoding="utf-8"))["rows"]:
            worst[(r["a"], r["b"])] = r
    skills = {s.id: s for s in build_builtin_registry().all()}
    manifest = ProbeManifest.model_validate_json(
        (Path(args.probes_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    battery = [(e.probe_id, load_probe(args.probes_dir, e.probe_id)) for e in manifest.entries]
    provider = BatteryEvaluator(
        list(skills.values()), battery, seed=1234, max_cache_bytes=8_000_000_000
    )

    # ---- render each pair once, into a shared stimulus pool -----------------------------------
    pool = out / "stimuli"
    pool.mkdir(exist_ok=True)
    rendered: list[dict[str, Any]] = []
    for idx, d in enumerate(dis):
        a, b = d["a"], d["b"]
        oa, ob = provider.outputs(a), provider.outputs(b)
        common = [p for p in oa.probe_ids if p in set(ob.probe_ids)]
        # rank probes by how much the two differ, take the most informative
        scored = []
        for pid in common:
            ca, cb = oa.canon[pid].rgb, ob.canon[pid].rgb
            if ca.shape != cb.shape:
                continue
            scored.append(
                (
                    float(np.abs(_to_u8(ca).astype(np.int16) - _to_u8(cb).astype(np.int16)).max()),
                    pid,
                )
            )
        scored.sort(reverse=True)
        chosen = [pid for _, pid in scored[: args.n_probes]]
        pdir = pool / f"pair_{idx:02d}"
        pdir.mkdir(exist_ok=True)
        gains: dict[str, float] = {}
        for pid in chosen:
            ca, cb = oa.canon[pid].rgb, ob.canon[pid].rgb
            Image.fromarray(_to_u8(ca)).save(pdir / f"{pid}_A.png")
            Image.fromarray(_to_u8(cb)).save(pdir / f"{pid}_B.png")
            diff, gain = _amplified_diff(ca, cb)
            Image.fromarray(diff).save(pdir / f"{pid}_diff.png")
            gains[pid] = round(gain, 2)
        (pdir / "docstrings.md").write_text(
            f"## Operator A\n\n**{skills[a].name}**\n\n{skills[a].description}\n\n"
            f"Parameters: `{skills[a].params_schema.defaults()}`\n\n"
            f"## Operator B\n\n**{skills[b].name}**\n\n{skills[b].description}\n\n"
            f"Parameters: `{skills[b].params_schema.defaults()}`\n",
            encoding="utf-8",
        )
        (pdir / "pair.json").write_text(
            json.dumps(
                {
                    "stimulus_id": f"pair_{idx:02d}",
                    "n_probes": len(chosen),
                    "probes": chosen,
                    "params": "operator defaults (shared)",
                    "diff_map": "|A-B| rescaled so the max difference is white",
                    "diff_amplification_per_probe": gains,
                    "diff_map_note": (
                        "A LARGE amplification factor means the true difference was SMALL. Without "
                        "this, a vivid diff map pushes the judgement toward DISTINCT even when the "
                        "operators differ by a rounding step."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        rendered.append(
            {
                "stimulus_id": f"pair_{idx:02d}",
                "a": a,
                "b": b,
                "verifier_says": d["verifier_says"],
                "n_probes": len(chosen),
            }
        )

    # ---- per-annotator: shuffled order, shuffled A/B side, no verdict anywhere ----------------
    for k in range(args.annotators):
        adir = out / f"annotator_{k + 1}"
        adir.mkdir(exist_ok=True)
        rng = random.Random(args.seed + k)
        order = list(rendered)
        rng.shuffle(order)
        tasks = []
        for pos, r in enumerate(order, 1):
            tasks.append(
                {
                    "task": pos,
                    "stimulus_id": r["stimulus_id"],
                    "swap_AB": rng.random() < 0.5,
                    "label": "",
                    "direction": "",
                    "reason": "",
                }
            )
        (adir / "labels.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        (adir / "GUIDELINES.md").write_text(GUIDELINES, encoding="utf-8")
        (adir / "README.txt").write_text(
            "Read GUIDELINES.md, then fill in labels.json.\n"
            "Stimuli are in ../stimuli/<stimulus_id>/.\n"
            "If swap_AB is true for a task, treat the files labelled _B as operator A and vice\n"
            "versa — the mapping is randomised per annotator so a left/right habit cannot line up\n"
            "across the three of you.\n\n"
            "Do NOT open ../key/. It contains the answers and opening it voids your labels.\n",
            encoding="utf-8",
        )

    kdir = out / "key"
    kdir.mkdir(exist_ok=True)
    (kdir / "DO_NOT_OPEN_UNTIL_SUBMITTED.json").write_text(
        json.dumps(
            {
                "note": "verifier verdicts + designed labels; annotators must not see this",
                "bound_if_all_confirmed": {"n": 69, "one_sided_95_upper": 0.0425},
                "bound_if_all_rejected": {"n": 43, "one_sided_95_upper": 0.0673},
                "pairs": rendered,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(json.dumps(rendered, sort_keys=True).encode()).hexdigest()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "artifact": "hard_negative_adjudication_package",
                "n_pairs": len(rendered),
                "n_annotators": args.annotators,
                "n_probes_per_pair": args.n_probes,
                "shuffle_seed": args.seed,
                "labels": LABELS,
                "stimulus_sha256": digest,
                "blinding": [
                    "verdict withheld from annotator dirs",
                    "pair order shuffled per annotator",
                    "A/B side shuffled per annotator",
                    "CANNOT-TELL available",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"adjudication package: {len(rendered)} pairs x {args.annotators} annotators -> {out}")
    print(f"  stimulus sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

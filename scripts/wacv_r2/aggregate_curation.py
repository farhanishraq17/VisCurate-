#!/usr/bin/env python
"""Table 1, rebuilt from the raw per-episode records with correct uncertainty.

WHY NOT `viscurate phase8`. Phase 8 aggregates ``StudyPoint`` rows, and ``downstream_success`` is a
REQUIRED field on that model — it comes from Phase 7 (``build-queries`` + ``run-downstream``),
which has never been run: there is no ``data/queries`` and no downstream result anywhere in the
repo. Supplying Phase 8 with invented success values to make its Pareto front and
construct-validity correlation render would be fabrication, so Phase 8's outputs are reported as
BLOCKED and this script produces the part of Table 1 that the real artifacts do support: the
action-level scores already computed per episode.

WHAT IT FIXES relative to the shipped ``summary.json`` files:

* **Uncertainty is clustered by corruption instance, never by pair or action.** Episodes on the
  same instance share a library and a defect draw, so they are dependent; a naive per-episode
  interval understates variance (Experiments/README.md §rule 5). Intervals here are
  instance-level bootstrap percentile intervals.
* **Every cell reports its denominator**, and a cell whose episode count differs from the
  others is flagged rather than averaged silently — the archived gpt-5.5 cell has 259 of 300
  episodes and its missing 41 are exactly the hardest cells, which biases its mean upward.
* **A per-rho breakdown**, so a truncated cell's bias is visible instead of hidden in one mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = ("f1", "precision", "recall", "intrinsic_score", "action_cost")


def bootstrap_ci(
    by_cluster: dict[str, list[float]], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI resampling CLUSTERS (instances) with replacement, not individual episodes."""
    keys = list(by_cluster)
    if not keys:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        vals: list[float] = []
        for _ in keys:
            vals.extend(by_cluster[keys[rng.randrange(len(keys))]])
        if vals:
            means.append(st.mean(vals))
    if not means:
        return (float("nan"), float("nan"))
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return (lo, hi)


def load_cell(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for score in sorted(root.glob("*/score.json")):
        rec = json.loads(score.read_text(encoding="utf-8"))
        name = score.parent.name
        # instance names look like rho090_uniform_seed1234_mixed
        parts = name.split("_")
        rho = int(parts[0].removeprefix("rho")) / 100.0
        rec["_instance"] = name
        rec["_rho"] = rho
        rec["_mode"] = parts[-1]
        rec["_composition"] = "_".join(parts[1:-2])
        out[name] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", action="append", required=True, help="label=path, repeatable")
    ap.add_argument("--out", default="results/wacv_r2/table1")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.cell:
        label, _, path = spec.partition("=")
        recs = load_cell(Path(path))
        if recs:
            cells[label] = recs
        else:
            print(f"  [skip] {label}: no score.json under {path}")

    counts = {k: len(v) for k, v in cells.items()}
    n_max = max(counts.values()) if counts else 0
    rows: list[dict[str, Any]] = []
    for label, recs in cells.items():
        row: dict[str, Any] = {
            "model": label,
            "n_episodes": len(recs),
            "complete": len(recs) == n_max,
            "n_expected": n_max,
        }
        for metric in METRICS:
            vals = [float(r[metric]) for r in recs.values() if metric in r]
            by_inst = defaultdict(list)
            for r in recs.values():
                if metric in r:
                    by_inst[r["_instance"]].append(float(r[metric]))
            lo, hi = bootstrap_ci(by_inst, n_boot=args.n_boot)
            row[f"mean_{metric}"] = st.mean(vals) if vals else None
            row[f"ci95_{metric}"] = [round(lo, 4), round(hi, 4)]
        rows.append(row)
    rows.sort(key=lambda r: -(r["mean_f1"] or 0))

    # per-rho breakdown, so a truncated cell's bias is visible
    per_rho: list[dict[str, Any]] = []
    for label, recs in cells.items():
        by_rho: dict[float, list[float]] = defaultdict(list)
        for r in recs.values():
            by_rho[r["_rho"]].append(float(r["f1"]))
        for rho in sorted(by_rho):
            per_rho.append(
                {
                    "model": label,
                    "rho": rho,
                    "n": len(by_rho[rho]),
                    "mean_f1": round(st.mean(by_rho[rho]), 4),
                }
            )

    (out_dir / "table1.json").write_text(
        json.dumps({"rows": rows, "per_rho": per_rho}, indent=2), encoding="utf-8"
    )
    with (out_dir / "table1.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    L = ["# Table 1 — curation-agent grid, rebuilt from raw per-episode records", ""]
    L += [
        f"- cells: **{len(cells)}** · episodes per complete cell: **{n_max}**",
        "- intervals: **instance-clustered bootstrap** percentile 95% CIs "
        f"({args.n_boot} resamples), not the normal approximation the shipped module uses",
        "- Phase-8 Pareto / construct-validity / vision-matters: **BLOCKED** — they need "
        "`downstream_success`, which comes from Phase 7 (`build-queries` + `run-downstream`); no "
        "Phase-7 artifact exists in this repo, and inventing success values to make those figures "
        "render would be fabrication.",
        "",
        "| model | n | complete | mean F1 [95% CI] | mean precision | mean recall | "
        "mean intrinsic | mean action cost |",
        "|---|---:|:--:|---|---|---|---|---|",
    ]
    for r in rows:
        c = "yes" if r["complete"] else f"**NO ({r['n_episodes']}/{r['n_expected']})**"

        def f(m: str, rr: dict[str, Any] = r) -> str:
            v = rr.get(f"mean_{m}")
            return "—" if v is None else f"{v:.4f}"

        ci = r["ci95_f1"]
        L.append(
            f"| `{r['model']}` | {r['n_episodes']} | {c} | {f('f1')} [{ci[0]:.4f}, {ci[1]:.4f}] "
            f"| {f('precision')} | {f('recall')} | {f('intrinsic_score')} | {f('action_cost')} |"
        )
    L.append("")
    incomplete = [r for r in rows if not r["complete"]]
    if incomplete:
        L += [
            "> ⚠ **Incomplete cells above are not comparable to the complete ones at face value.** "
            "For the archived `gpt-5.5` cell the 41 absent episodes are exactly the hardest "
            "(all of ρ=1.0 plus the ρ=0.9 tail), so its mean is measured on an easier subset. The "
            "per-ρ table below is where that shows.",
            "",
        ]
    L += [
        "## Mean F1 by corruption rate ρ",
        "",
        "| model | " + " | ".join(f"ρ={r:.1f}" for r in sorted({p["rho"] for p in per_rho})) + " |",
        "|---" * (1 + len({p["rho"] for p in per_rho})) + "|",
    ]
    rhos = sorted({p["rho"] for p in per_rho})
    for label in [r["model"] for r in rows]:
        cellmap = {p["rho"]: p for p in per_rho if p["model"] == label}
        vals = []
        for rho in rhos:
            p = cellmap.get(rho)
            vals.append("—" if p is None else f"{p['mean_f1']:.3f} (n={p['n']})")
        L.append(f"| `{label}` | " + " | ".join(vals) + " |")
    L.append("")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"table1 -> {out_dir}/report.md ({len(cells)} cells)")
    for r in rows:
        print(f"  {r['model']:24} n={r['n_episodes']:>3} F1={r['mean_f1']:.4f} {r['ci95_f1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

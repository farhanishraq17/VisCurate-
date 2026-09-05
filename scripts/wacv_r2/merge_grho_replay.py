#!/usr/bin/env python
"""Aggregate the `G_rho` replay shards into the verifier-validation report.

Reports exactly what NEXT_ACTIONS.md §1 asks for: per-relation P/R/F1 with support, the mergeable
binary decision (now with a POPULATED positive class), the full confusion matrix including
UNCERTAIN, a breakdown by corruption rate, exact Clopper-Pearson intervals on every rate, and
bootstrap intervals that resample CORRUPTION INSTANCES rather than pairs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from viscurate.equivalence.relations import Relation

ORDER = [r.value for r in Relation]
MERGEABLE = {Relation.EXACT.value, Relation.PERCEPTUAL.value}


def cp(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta

    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def boot_by_instance(
    per_inst: dict[str, list[bool]], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI resampling INSTANCES. Pairs inside an instance share base skills and one
    defect draw, so they are dependent and pair-level resampling understates variance."""
    keys = list(per_inst)
    if not keys:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        vals: list[bool] = []
        for _ in keys:
            vals.extend(per_inst[keys[rng.randrange(len(keys))]])
        if vals:
            means.append(sum(vals) / len(vals))
    means.sort()
    return (means[int(0.025 * len(means))], means[min(len(means) - 1, int(0.975 * len(means)))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/wacv_r2/grho_replay")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.dir)
    out = Path(args.out) if args.out else d
    shards = sorted(d.glob("shard*.json"))
    if not shards:
        raise SystemExit(f"no shard files under {d}")

    rows: list[dict[str, Any]] = []
    repair = Counter()
    meta: list[dict[str, Any]] = []
    # ★ DEDUP BY (instance, a, b). Shards are re-run after OOM kills and the sweeper re-runs any
    # missing instance, so the same pair can legitimately appear in more than one shard file.
    # Without this, a re-run silently DOUBLE-COUNTS pairs and inflates every support figure.
    # Verification is deterministic (fixed seed, carried-over thresholds), so keeping the first
    # occurrence is safe; a disagreement between copies is a determinism bug and is reported.
    seen: dict[tuple[str, str, str], str] = {}
    row_index: dict[tuple[str, str, str], int] = {}
    dupes = 0
    repaired = 0
    conflicts: list[dict[str, Any]] = []
    for sp in shards:
        p = json.loads(sp.read_text(encoding="utf-8"))
        for r in p["rows"]:
            key = (r.get("instance", ""), r["a"], r["b"])
            if key in seen:
                dupes += 1
                prev = seen[key]
                # A re-run exists precisely to REPAIR an ERROR row, so a successful verdict must
                # win over a failed one; keeping the first occurrence would make error-repair a
                # no-op. Verification is deterministic, so two successful copies can only differ
                # if determinism is broken -- which is recorded as a conflict.
                if prev == "ERROR" and r["pred"] != "ERROR":
                    idx = row_index[key]
                    rows[idx] = r
                    seen[key] = r["pred"]
                    repaired += 1
                elif prev != "ERROR" and r["pred"] != "ERROR" and prev != r["pred"]:
                    conflicts.append({"key": list(key), "pred_a": prev, "pred_b": r["pred"]})
                continue
            seen[key] = r["pred"]
            row_index[key] = len(rows)
            rows.append(r)
        for k, v in p["key_repair"].items():
            if isinstance(v, int):
                repair[k] += v
        meta.append(
            {k: p.get(k) for k in ("shard", "n_instances", "n_pairs", "wall_seconds", "complete")}
        )
    if dupes:
        print(
            f"  dedup: dropped {dupes} duplicate pair rows from re-run shards; "
            f"{repaired} ERROR rows repaired by a later successful run; "
            f"{len(conflicts)} disagreed on the verdict",
            flush=True,
        )
    errs = [r for r in rows if r["pred"] == "ERROR"]
    ok = [r for r in rows if r["pred"] != "ERROR"]
    instances = sorted({r["instance"] for r in ok})

    # ---- mergeable decision, the number that could not exist on G_0 --------------------------
    tp = sum(1 for r in ok if r["truth_mergeable"] and r["pred_mergeable"])
    fp = sum(1 for r in ok if not r["truth_mergeable"] and r["pred_mergeable"])
    fn = sum(1 for r in ok if r["truth_mergeable"] and not r["pred_mergeable"])
    tn = sum(1 for r in ok if not r["truth_mergeable"] and not r["pred_mergeable"])
    n_pos, n_pred = tp + fn, tp + fp
    recall = tp / n_pos if n_pos else None
    prec = tp / n_pred if n_pred else None
    f1 = (2 * prec * recall / (prec + recall)) if (prec and recall) else None
    rec_by_inst = defaultdict(list)
    for r in ok:
        if r["truth_mergeable"]:
            rec_by_inst[r["instance"]].append(bool(r["pred_mergeable"]))
    rboot = boot_by_instance(rec_by_inst)
    rcp = cp(tp, n_pos)
    fdr = fp / n_pred if n_pred else None

    # ---- per-relation ------------------------------------------------------------------------
    per_rel = {}
    for rel in ORDER:
        sup = [r for r in ok if r["truth"] == rel]
        pred = [r for r in ok if r["pred"] == rel]
        h = sum(1 for r in sup if r["pred"] == rel)
        p = h / len(pred) if pred else None
        rc = h / len(sup) if sup else None
        per_rel[rel] = {
            "support": len(sup),
            "n_pred": len(pred),
            "hits": h,
            "precision": p,
            "recall": rc,
            "f1": (2 * p * rc / (p + rc)) if (p and rc) else None,
            "recall_ci95": cp(h, len(sup)) if sup else None,
        }

    # ---- confusion + rho ---------------------------------------------------------------------
    conf = Counter((r["truth"], r["pred"]) for r in ok)
    by_rho: dict[float, dict[str, Any]] = {}
    for rho in sorted({r["rho"] for r in ok}):
        sub = [r for r in ok if r["rho"] == rho]
        pos = [r for r in sub if r["truth_mergeable"]]
        by_rho[rho] = {
            "n": len(sub),
            "mergeable_support": len(pos),
            "recall": (sum(1 for r in pos if r["pred_mergeable"]) / len(pos)) if pos else None,
            "false_merges": sum(1 for r in sub if not r["truth_mergeable"] and r["pred_mergeable"]),
        }

    payload = {
        "artifact": "grho_replay",
        "n_instances": len(instances),
        "n_pairs_scored": len(ok),
        "n_errors": len(errs),
        "shards": meta,
        "key_repair_totals": dict(repair),
        "dedup": {
            "duplicate_rows_dropped": dupes,
            "error_rows_repaired": repaired,
            "verdict_conflicts": conflicts[:20],
            "n_conflicts": len(conflicts),
        },
        "scoring_set": "explicitly-labelled pairs of the REPAIRED key only; precision figures are "
        "within that set. The distinct-sea false-merge rate is G_0's 0/926 and the "
        "expanded slice's 0/69 and is NOT re-estimated here.",
        "mergeable": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": n_pos,
            "recall": recall,
            "recall_ci95_clopper_pearson": [round(rcp[0], 4), round(rcp[1], 4)],
            "recall_ci95_bootstrap_by_instance": [round(rboot[0], 4), round(rboot[1], 4)],
            "precision": prec,
            "f1": f1,
            "fdr_merge": fdr,
        },
        "per_relation": per_rel,
        "confusion": {f"{t}->{p}": n for (t, p), n in sorted(conf.items())},
        "by_rho": by_rho,
    }
    (out / "grho_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def f(x: Any, nd: int = 4) -> str:
        return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    L = ["# `G_rho` replay — validating the verifier on the relations that authorise deletion", ""]
    L += [
        f"- instances: **{len(instances)}** (stratified, 4 per ρ level, seed 1234) · pairs scored: "
        f"**{len(ok)}** · errors: {len(errs)}",
        "- thresholds carried over from the Stage-1 calibration; **nothing re-calibrated**",
        f"- answer key **repaired first**: {repair.get('labelled', 0)} labelled pairs after "
        f"materialising EXACT-clone inheritance; {repair.get('excluded_perceptual_inherited', 0)} "
        "PERCEPTUAL-clone-inherited pairs **excluded** rather than guessed",
        "- scoring set: explicitly-labelled pairs only, so precision below is *within that set*. "
        "The distinct-sea false-merge rate remains `G_0`'s 0/926 and the expanded slice's 0/69.",
        "",
        "## ★ The mergeable decision — the positive class `G_0` could not provide",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| mergeable support (EXACT ∪ PERCEPTUAL) | **{n_pos}** |",
        f"| **merge recall** | **{f(recall)}** — exact 95% CI [{rcp[0]:.4f}, {rcp[1]:.4f}]; "
        f"instance-bootstrap [{rboot[0]:.4f}, {rboot[1]:.4f}] |",
        f"| precision (within labelled set) | {f(prec)} |",
        f"| F1 | {f(f1)} |",
        f"| FDR on merges | {f(fdr)} |",
        f"| tp / fp / fn / tn | {tp} / {fp} / {fn} / {tn} |",
        "",
        "**This is the number R3 asked for.** Every prior run reported `0.000/0.000/0.000` here "
        "because the class was empty, which reads as measured failure rather than absent data.",
        "",
        "## Per-relation",
        "",
        "| relation | support | predicted | recall [95% CI] | precision | F1 |",
        "|---|---:|---:|---|---|---|",
    ]
    for rel in ORDER:
        m = per_rel[rel]
        ci = m["recall_ci95"]
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        L.append(
            f"| {rel} | {m['support']} | {m['n_pred']} | {f(m['recall'])} {ci_s} "
            f"| {f(m['precision'])} | {f(m['f1'])} |"
        )
    L += [
        "",
        "## By corruption rate ρ",
        "",
        "| ρ | pairs | mergeable support | merge recall | false merges |",
        "|---|---:|---:|---|---:|",
    ]
    for rho, m in by_rho.items():
        L.append(
            f"| {rho:.1f} | {m['n']} | {m['mergeable_support']} | {f(m['recall'])} "
            f"| {m['false_merges']} |"
        )
    L += [
        "",
        "## Confusion (truth → predicted)",
        "",
        "| truth \\ pred | " + " | ".join(ORDER) + " |",
        "|---" * (len(ORDER) + 1) + "|",
    ]
    for t in ORDER:
        L.append(f"| **{t}** | " + " | ".join(str(conf.get((t, p), 0)) for p in ORDER) + " |")
    L.append("")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(
        f"grho replay: {len(ok)} pairs, mergeable support {n_pos}, "
        f"recall {f(recall)} -> {out}/report.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

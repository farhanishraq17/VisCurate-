#!/usr/bin/env python
"""Merge the hard-negative shards into one slice result with exact binomial bounds.

The union over shards must be exactly the slice — the merge asserts that rather than trusting it,
because a silently missing shard would inflate the safety claim by shrinking the denominator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta

    if n == 0:
        return (0.0, 1.0)
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (low, high)


def upper_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    from scipy.stats import beta

    if n == 0:
        return 1.0
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/wacv_r2/hardneg69")
    ap.add_argument("--expect", type=int, default=69)
    args = ap.parse_args()
    d = Path(args.dir)
    shards = sorted(d.glob("shard*.json"))
    if not shards:
        raise SystemExit(f"no shard files under {d}")
    rows: list[dict] = []
    by_key: dict[tuple[str, str], dict] = {}
    disagreements: list[dict] = []
    metas = []
    for sp in shards:
        payload = json.loads(sp.read_text(encoding="utf-8"))
        metas.append({k: payload[k] for k in ("shard", "shards", "n_pairs", "wall_seconds")})
        for r in payload["rows"]:
            key = (r["a"], r["b"])
            prev = by_key.get(key)
            if prev is None:
                by_key[key] = r
                rows.append(r)
                continue
            # Shards overlap (run_benchmark always scores the whole designed set), so the same pair
            # is verified independently several times. Identical inputs must give identical
            # verdicts; a mismatch means the pipeline is not deterministic and no rate below is
            # trustworthy, so it is surfaced rather than resolved by taking the first one.
            if prev["output_relation"] != r["output_relation"]:
                disagreements.append(
                    {
                        "pair": list(key),
                        "verdicts": sorted({prev["output_relation"], r["output_relation"]}),
                    }
                )
    n = len(rows)
    fm = sum(1 for r in rows if r["output_mergeable"])
    lo, hi = clopper_pearson(fm, n)
    hi1 = upper_one_sided(fm, n)
    complete = n == args.expect
    by_rel: dict[str, int] = {}
    for r in rows:
        by_rel[r["output_relation"]] = by_rel.get(r["output_relation"], 0) + 1

    out = {
        "artifact": "hard_negative_slice_expanded",
        "n_pairs": n,
        "n_expected": args.expect,
        "complete": complete,
        "false_merges": fm,
        "false_merge_rate": fm / n if n else None,
        "ci95_two_sided": [round(lo, 5), round(hi, 5)],
        "upper95_one_sided": round(hi1, 5),
        "verdicts": by_rel,
        "shards": metas,
        "n_shard_overlap_disagreements": len(disagreements),
        "shard_disagreements": disagreements,
    }
    (d / "slice_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    L = ["# Expanded hard-negative slice — output-grounded verifier", ""]
    L += [
        f"- pairs scored: **{n}** of {args.expect} expected"
        + ("" if complete else "  ⚠ **INCOMPLETE — a missing shard shrinks the denominator**"),
        f"- **false merges: {fm}/{n}**"
        + (f" = {fm / n:.4f}" if n else "")
        + f", exact 95% CI [{lo:.4f}, {hi:.4f}], **one-sided 95% upper bound {hi1:.4f}**",
        "- thresholds carried over from the Stage-1 calibration; **nothing recalibrated here**",
        f"- cross-shard determinism check: **{len(disagreements)} disagreement(s)** over "
        f"pairs verified independently by more than one shard"
        + (
            ""
            if not disagreements
            else "  ⚠ **the pipeline is not deterministic; no rate here is trustworthy**"
        ),
        "",
        f"For comparison the 6-pair slice admits a one-sided upper bound of "
        f"{upper_one_sided(0, 6):.3f} even at 0/6, which is why the expansion was needed.",
        "",
        "## Verdicts over the slice",
        "",
        "| relation | n |",
        "|---|---:|",
    ]
    for rel, cnt in sorted(by_rel.items(), key=lambda kv: -kv[1]):
        L.append(f"| {rel} | {cnt} |")
    L.append("")
    if fm:
        L += [
            "## Every false merge, enumerated",
            "",
            "| a | b | verdict | reason |",
            "|---|---|---|---|",
        ]
        for r in rows:
            if r["output_mergeable"]:
                L.append(
                    f"| `{r['a']}` | `{r['b']}` | {r['output_relation']} "
                    f"| {r.get('reason', '')[:110]} |"
                )
        L.append("")
    (d / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"slice: {fm}/{n} false merges, one-sided 95% upper {hi1:.4f} -> {d}/report.md")
    if not complete:
        print(f"  ⚠ INCOMPLETE: {n} of {args.expect}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

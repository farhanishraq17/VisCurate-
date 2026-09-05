#!/usr/bin/env python
"""NEXT_ACTIONS.md §2 — score the returned adjudications and derive the safety bound.

Run this once all three annotators have filled their `labels.json`. It un-shuffles each
annotator's order and un-swaps their A/B assignment, computes agreement, resolves a consensus, and
reports **both** bounds — conditional and unconditional — as the protocol requires.

κ comes from the repo's own `benchmark/human_review.py` (`cohen_kappa` / `fleiss_kappa`) rather
than a reimplementation. What is NOT reused is that module's label loader: it silently drops any
label that is not a `Relation`, which would swallow **`CANNOT-TELL`** — the outcome the protocol
specifically requires be reported. Here it is a first-class category with its own rate, and κ is
computed both including and excluding it, because treating an abstention as a category and
treating it as missing data give different agreement numbers and only reporting the flattering one
would be a choice, not a measurement.

THE BOUND. A pair only remains a hard negative if consensus says `DISTINCT`. Any pair the
annotators call something else is not a hard negative and leaves the slice. `CANNOT-TELL` and
unresolved disagreements are **excluded** — the conservative direction, since dropping a pair
shrinks n and *widens* the bound.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from viscurate.benchmark.human_review import cohen_kappa, fleiss_kappa
from viscurate.equivalence.relations import Relation

CANNOT_TELL = "CANNOT-TELL"


def upper_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    from scipy.stats import beta

    if n == 0:
        return 1.0
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta

    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="results/wacv_r2/adjudication")
    ap.add_argument(
        "--slice-agreed",
        type=int,
        default=43,
        help="hard negatives the verifier already agrees are DISTINCT",
    )
    ap.add_argument("--observed-false-merges", type=int, default=0)
    args = ap.parse_args()
    pkg = Path(args.package)
    key = json.loads((pkg / "key" / "DO_NOT_OPEN_UNTIL_SUBMITTED.json").read_text(encoding="utf-8"))
    by_stim = {r["stimulus_id"]: r for r in key["pairs"]}

    # ---- collect, un-shuffled ---------------------------------------------------------------
    per_stim: dict[str, dict[str, str]] = {}
    annotators: list[str] = []
    for adir in sorted(pkg.glob("annotator_*")):
        tasks = json.loads((adir / "labels.json").read_text(encoding="utf-8"))
        filled = [t for t in tasks if str(t.get("label", "")).strip()]
        if not filled:
            print(f"  [skip] {adir.name}: labels.json is unfilled")
            continue
        annotators.append(adir.name)
        for t in tasks:
            lab = str(t.get("label", "")).strip().upper()
            if not lab:
                continue
            per_stim.setdefault(t["stimulus_id"], {})[adir.name] = lab
    if len(annotators) < 2:
        raise SystemExit(
            f"need at least 2 filled annotator files, found {len(annotators)}. "
            "The package is ready; this scorer runs once they are returned."
        )
    print(f"annotators: {len(annotators)} ({', '.join(annotators)})")

    complete = [s for s, d in per_stim.items() if len(d) == len(annotators)]
    print(f"stimuli labelled by all {len(annotators)}: {len(complete)}/{len(by_stim)}")

    # ---- agreement --------------------------------------------------------------------------
    cannot = Counter()
    for s in complete:
        for a, lab in per_stim[s].items():
            if lab == CANNOT_TELL:
                cannot[a] += 1
    n_cells = len(complete) * len(annotators)
    ct_rate = sum(cannot.values()) / n_cells if n_cells else 0.0

    def as_rel(lab: str) -> Relation | None:
        try:
            return Relation(lab)
        except ValueError:
            return None

    kappas: dict[str, Any] = {}
    # κ EXCLUDING cannot-tell: only stimuli every annotator gave a relation to
    rel_only = [s for s in complete if all(as_rel(per_stim[s][a]) for a in annotators)]
    if len(rel_only) >= 2:
        pair_k = {}
        for a, b in combinations(annotators, 2):
            pair_k[f"{a}|{b}"] = cohen_kappa(
                [as_rel(per_stim[s][a]) for s in rel_only],  # type: ignore[misc]
                [as_rel(per_stim[s][b]) for s in rel_only],  # type: ignore[misc]
            )
        kappas["cohen_pairwise_excluding_cannot_tell"] = pair_k
        if len(annotators) >= 3:
            kappas["fleiss_excluding_cannot_tell"] = fleiss_kappa(
                [[as_rel(per_stim[s][a]) for a in annotators] for s in rel_only]  # type: ignore[misc]
            )
        kappas["n_items_excluding_cannot_tell"] = len(rel_only)
    else:
        kappas["note"] = "too few fully-relation-labelled stimuli for κ"

    # ---- consensus and the resulting slice ---------------------------------------------------
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    for s in complete:
        votes = Counter(per_stim[s].values())
        top, n_top = votes.most_common(1)[0]
        if n_top > len(annotators) / 2 and top != CANNOT_TELL:
            resolved[s] = top
        else:
            unresolved.append(s)
    confirmed = [s for s, lab in resolved.items() if lab == Relation.DISTINCT.value]
    rejected = [s for s, lab in resolved.items() if lab != Relation.DISTINCT.value]

    n = args.slice_agreed + len(confirmed)
    k = args.observed_false_merges
    lo, hi = clopper_pearson(k, n)
    u1 = upper_one_sided(k, n)
    n_all, n_none = args.slice_agreed + len(by_stim), args.slice_agreed
    payload = {
        "artifact": "hard_negative_adjudication_result",
        "annotators": annotators,
        "n_stimuli": len(by_stim),
        "n_labelled_by_all": len(complete),
        "cannot_tell_rate": round(ct_rate, 4),
        "cannot_tell_per_annotator": dict(cannot),
        "kappa": kappas,
        "consensus": {
            "confirmed_DISTINCT": confirmed,
            "rejected": rejected,
            "unresolved_or_cannot_tell": unresolved,
        },
        "slice": {
            "agreed_by_verifier": args.slice_agreed,
            "confirmed_by_annotators": len(confirmed),
            "n": n,
            "false_merges": k,
            "ci95_two_sided": [round(lo, 5), round(hi, 5)],
            "upper95_one_sided": round(u1, 5),
            "clears_5pct": bool(u1 <= 0.05),
        },
        "brackets": {
            "if_all_26_confirmed": {
                "n": n_all,
                "upper95_one_sided": round(upper_one_sided(k, n_all), 5),
            },
            "if_all_26_rejected": {
                "n": n_none,
                "upper95_one_sided": round(upper_one_sided(k, n_none), 5),
            },
        },
    }
    (pkg / "adjudication_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    L = ["# Hard-negative adjudication — result", ""]
    L += [
        f"- annotators: **{len(annotators)}** · stimuli labelled by all: "
        f"**{len(complete)}/{len(by_stim)}**",
        f"- **`CANNOT-TELL` rate: {ct_rate:.1%}** ({sum(cannot.values())} of {n_cells} labels) — "
        "reported as its own outcome, never folded into a relation",
        "- κ (pairwise Cohen, excluding CANNOT-TELL): "
        + json.dumps(kappas.get("cohen_pairwise_excluding_cannot_tell", kappas.get("note"))),
        f"- Fleiss κ: {kappas.get('fleiss_excluding_cannot_tell', '—')}",
        "",
        "## The bound",
        "",
        "| counting | n | false merges | one-sided 95% upper |",
        "|---|---:|---:|---|",
        f"| **adjudicated** | **{n}** | {k} | **{u1:.4f}** "
        f"{'✓ clears 5%' if u1 <= 0.05 else '✗ over 5%'} |",
        f"| bracket: all 26 confirmed | {n_all} | {k} | {upper_one_sided(k, n_all):.4f} |",
        f"| bracket: all 26 rejected | {n_none} | {k} | {upper_one_sided(k, n_none):.4f} |",
        "",
        f"Consensus: **{len(confirmed)}** confirmed `DISTINCT` (stay in the slice), "
        f"**{len(rejected)}** re-labelled (leave the slice), "
        f"**{len(unresolved)}** unresolved or `CANNOT-TELL` (excluded — which widens the bound, "
        "the conservative direction).",
        "",
    ]
    if rejected:
        L += [
            "### Pairs the annotators removed from the slice",
            "",
            "| pair | consensus | verifier had said |",
            "|---|---|---|",
        ]
        for s in sorted(rejected):
            r = by_stim[s]
            L.append(f"| `{r['a']}` vs `{r['b']}` | {resolved[s]} | {r['verifier_says']} |")
        L.append("")
    (pkg / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"adjudication: n={n}, one-sided 95% upper {u1:.4f} -> {pkg}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

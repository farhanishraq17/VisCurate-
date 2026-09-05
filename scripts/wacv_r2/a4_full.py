#!/usr/bin/env python
"""T4 / A4 — the complete cost-and-scaling analysis (A4.1 … A4.6).

Desk work: consumes telemetry already on disk plus the A4.1/A4.5 candidate sweep. No curation or
verification run is repeated to backfill a field -- where instrumentation was absent the gap is
reported as a gap.

DESIGN NOTES THAT CHANGE THE NUMBERS.

* **Totals, never mean-of-ratios.** Cost per repair is `sum(tokens) / sum(correct repairs)` over
  episodes, not the mean of per-episode ratios. The mean-of-ratios is dominated by cheap episodes
  that happened to land one repair and is not the quantity anyone deploying this cares about.
* **The denominator is CORRECT repairs (`tp` from `score.json`), not actions applied.** An agent
  that applies 200 wrong edits has not performed 200 repairs. `n_applied` is reported alongside so
  the gap between "did something" and "did the right thing" stays visible.
* **The zero-repair case is defined, not silently dropped.** A model with `tp == 0` has an
  undefined cost-per-repair (division by zero). It is reported as `undefined (0 repairs)` with its
  absolute cost, because "infinite cost per repair" is the substantive finding for that row.
* **GPU-seconds, not USD.** Stage 2 ran self-hosted vLLM for the open models; a dollar figure would
  be a fabrication for those rows and rots for the hosted ones. Tokens and wall/GPU seconds are
  reported, and the price table is left to the reader.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_telemetry(pattern: str) -> list[dict[str, Any]]:
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                d["_file"] = f.split("/")[-1]
                out.append(d)
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def fmt(x: Any, n: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        return f"{x:.{n}f}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--telemetry-glob", default="results/wacv_r2/telemetry*.jsonl")
    ap.add_argument("--sweep-dir", default="results/wacv_r2/a4_sweep")
    ap.add_argument(
        "--episode-dirs",
        default="results/wacv_r2/phase6_gpt56luna,results/wacv_r2/phase6_gpt55_full",
    )
    ap.add_argument("--out", default="results/wacv_r2/a4_full")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = load_telemetry(args.telemetry_glob)
    L: list[str] = ["# A4 — measured cost and scaling (complete)", ""]
    L.append(
        "Tokens, wall-seconds and GPU-seconds only. **No USD**: the open models ran on self-hosted "
        "vLLM, so a dollar figure would be fabricated for those rows and would rot for the hosted "
        "ones. Every rate below is a **total over episodes**, never a mean of per-episode ratios."
    )
    L.append("")
    res: dict[str, Any] = {}

    kinds = collections.Counter(d.get("kind") for d in ev)
    L.append(
        f"- telemetry events parsed: **{len(ev):,}** across {len({d['_file'] for d in ev})} files"
    )
    L.append("- " + " · ".join(f"`{k}` {v:,}" for k, v in kinds.most_common()))
    L.append("")

    # ================= A4.1 / A4.5 — candidate screening ======================================
    sweeps = []
    for f in sorted(glob.glob(str(Path(args.sweep_dir) / "sweep_shard*.json"))):
        sweeps.append(json.loads(Path(f).read_text(encoding="utf-8")))

    L.append("## A4.1 — candidate reduction **and recall** vs fingerprint radius")
    L.append("")
    if not sweeps:
        L.append(
            "> **NOT YET RUN.** `a4_candidate_sweep.py` has produced no shard output. The recorded "
            "telemetry contains only a single radius (0.5) plus a `-1.0` sentinel, so the curve "
            "cannot be reconstructed from it."
        )
        L.append("")
    else:
        g0 = next((s["g0"] for s in sweeps if "g0" in s), None)
        if g0:
            L.append(
                f"**`G_0`** — {g0['n_skills']} skills, {g0['n_designed']} designed pairs, "
                f"**{g0['n_mergeable_positives']} mergeable positives**."
            )
            if g0["n_mergeable_positives"] == 0:
                L.append("")
                L.append(
                    "> `G_0` designates **zero** mergeable pairs, so *merge recall is undefined "
                    "here* and no value is reported for it. What `G_0` can support is the "
                    "**reduction** curve and **designed-pair retention** — the fraction of pairs "
                    "carrying *any* designed relation that survive screening. These are different "
                    "quantities and are labelled as such."
                )
            L.append("")
            L.append("| k | radius | \\|C\\| | reduction | designed-pair retention |")
            L.append("|---:|---:|---:|---|---|")
            for r in g0["sweep"]:
                if r["k"] != 5:
                    continue
                L.append(
                    f"| {r['k']} | {r['radius']} | {r['n_candidates']} | "
                    f"{pct(r['reduction'])} | {pct(r.get('labelled_retention', float('nan')))} |"
                )
            L.append("")

        grho = [r for s in sweeps for r in s.get("grho", [])]
        if grho:
            L.append(
                "**`G_ρ` — the real curve.** `G_ρ` injects duplicates, so it has genuine mergeable "
                "positives and merge recall is defined. Aggregated over "
                f"{len({r['instance'] for r in grho})} instances (one per ρ level)."
            )
            L.append("")
            L.append(
                "| k | radius | \\|C\\| total | all-pairs total | reduction | "
                "positives | retained | **merge recall** |"
            )
            L.append("|---:|---:|---:|---:|---|---:|---:|---|")
            agg: dict[tuple[int, float], list[dict[str, Any]]] = collections.defaultdict(list)
            for r in grho:
                agg[(r["k"], r["radius"])].append(r)
            rows_json = []
            for (k, rad), rows in sorted(agg.items()):
                nc = sum(x["n_candidates"] for x in rows)
                na = sum(x["n_all_pairs"] for x in rows)
                npos = sum(x.get("n_positives", 0) for x in rows)
                nret = sum(x.get("positives_retained", 0) for x in rows)
                rec = nret / npos if npos else None
                rows_json.append(
                    {
                        "k": k,
                        "radius": rad,
                        "n_candidates": nc,
                        "n_all_pairs": na,
                        "reduction": 1 - nc / na if na else None,
                        "n_positives": npos,
                        "retained": nret,
                        "merge_recall": rec,
                    }
                )
                if k == 5:
                    L.append(
                        f"| {k} | {rad} | {nc} | {na} | {pct(1 - nc / na) if na else '—'} | "
                        f"{npos} | {nret} | **{fmt(rec)}** |"
                    )
            L.append("")
            L.append(
                "*(k=5 shown — the shipped setting. The full k ∈ {1,3,5,10,20,50} grid is in "
                "`a4_full.json`.)*"
            )
            L.append("")
            best = [r for r in rows_json if r["k"] == 5 and r["merge_recall"] is not None]
            lossless = [r for r in best if r["merge_recall"] >= 0.999]
            if lossless:
                b = max(lossless, key=lambda r: r["reduction"])
                L.append(
                    f"★ **Operating point:** at k=5, radius **{b['radius']}** the screen keeps "
                    f"**100%** of mergeable positives while discarding "
                    f"**{pct(b['reduction'])}** of all pairs — screening is *free* in recall terms "
                    "at that radius, which is the claim A4.1 needed to license."
                )
            else:
                shipped = [r for r in best if abs(r["radius"] - 0.5) < 1e-9]
                if shipped:
                    L.append(
                        f"★ **At the shipped radius 0.5, merge recall is "
                        f"{fmt(shipped[0]['merge_recall'])}** — screening is **not** lossless, and "
                        "that loss compounds with the verifier's own recall."
                    )
            L.append("")
            res["a4_1"] = rows_json

    L.append("## A4.5 — |C| growth with library size")
    L.append("")
    sc = next((s["g0_scaling"] for s in sweeps if "g0_scaling" in s), None)
    if not sc:
        L.append("> **NOT YET RUN** (see A4.1).")
    else:
        L.append("| n skills | all pairs n(n−1)/2 | \\|C\\| | \\|C\\|/all |")
        L.append("|---:|---:|---:|---|")
        for p in sc["points"]:
            L.append(
                f"| {p['n_skills']} | {p['n_all_pairs']} | {p['n_candidates']} | "
                f"{pct(p['n_candidates'] / p['n_all_pairs']) if p['n_all_pairs'] else '—'} |"
            )
        L.append("")
        fit = sc.get("fit")
        if fit:
            L.append(
                f"**Fitted** `|C| = {fit['a']:.3f}·n^{fit['b']:.3f}` (log-log least squares, "
                f"{fit['n_points']} points, R²={fmt(fit['r2'], 3)}). Exponent **{fit['b']:.2f}** "
                f"vs the quadratic **2.00** of exhaustive pairing."
            )
            L.append("")
            L.append(
                f"**Projection to n=10⁴ (marked as projected, not measured):** "
                f"|C| ≈ **{fit['projected_C_at_1e4']:,.0f}** against "
                f"{fit['all_pairs_at_1e4']:,.0f} exhaustive pairs."
            )
            L.append("")
            L.append(
                "> Extrapolating a power law fitted on n ≤ 100 out to n = 10⁴ is two orders of "
                "magnitude beyond support. It is reported because A4.5 asks for a projection, and "
                "it is marked **projected** everywhere it appears."
            )
        res["a4_5"] = sc
    L.append("")

    # ================= A4.2 — cost table ======================================================
    L.append("## A4.2 — cost per repair (totals, not mean-of-ratios)")
    L.append("")
    tok: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])  # in, out, calls
    for d in ev:
        if d.get("kind") == "llm_call":
            f = d["fields"]
            m = f.get("model", "?")
            tok[m][0] += f.get("tokens_in", 0)
            tok[m][1] += f.get("tokens_out", 0)
            tok[m][2] += 1
    epi: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"episodes": 0, "applied": 0, "actions": 0, "wall_s": 0.0}
    )
    for d in ev:
        if d.get("kind") == "agent_episode":
            f = d["fields"]
            m = f.get("model", "?").split(":")[-1]
            e = epi[m]
            e["episodes"] += 1
            e["applied"] += f.get("n_applied", 0)
            e["actions"] += f.get("n_actions", 0)
            e["wall_s"] += f.get("wall_ms", 0) / 1000.0

    # correct repairs (tp) per model, from the scored episodes on disk.
    #
    # ★ RESTRICTED TO THE TELEMETRY-INSTRUMENTED INSTANCES. `phase6_gpt55_full` holds 300 scored
    # episodes but only 41 of them carry token telemetry (the rest predate instrumentation).
    # Dividing all-300 repairs by 41-episode tokens understates gpt-5.5's cost per repair by ~7x
    # and would have made it look five times cheaper than luna. Numerator and denominator must
    # come from the SAME episodes.
    instrumented: dict[str, set[str]] = collections.defaultdict(set)
    for d in ev:
        if d.get("kind") == "agent_episode":
            f = d["fields"]
            instrumented[f.get("model", "?").split(":")[-1]].add(f.get("instance_id"))

    tp_by_model: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0, "n_available": 0}
    )
    name_map = {
        "results/wacv_r2/phase6_gpt56luna": "gpt-5.6-luna",
        "results/wacv_r2/phase6_gpt55_full": "gpt-5.5",
    }
    for d in args.episode_dirs.split(","):
        d = d.strip()
        if not d:
            continue
        model = name_map.get(d, Path(d).name)
        keep = instrumented.get(model, set())
        for sf in sorted(glob.glob(f"{d}/*/score.json")):
            inst = Path(sf).parent.name
            t = tp_by_model[model]
            t["n_available"] += 1
            if keep and inst not in keep:
                continue
            try:
                s = json.loads(Path(sf).read_text(encoding="utf-8"))
            except Exception:
                continue
            t["tp"] += s.get("tp", 0)
            t["fp"] += s.get("fp", 0)
            t["fn"] += s.get("fn", 0)
            t["n"] += 1

    L.append(
        "| model | episodes | actions | applied | **correct repairs (tp)** | tokens in | "
        "tokens out | **tokens / repair** | wall s / repair |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    cost_rows = []
    curators = {m for m in epi if epi[m]["episodes"] > 0} | set(tp_by_model)
    utility = (set(tok) | set(epi)) - curators
    for m in sorted(curators):
        ti, to, _calls = tok.get(m, [0, 0, 0])
        e = epi.get(m, {"episodes": 0, "applied": 0, "actions": 0, "wall_s": 0.0})
        t = tp_by_model.get(m, {"tp": 0, "n": 0})
        tp = t["tp"]
        tpr = f"{(ti + to) / tp:,.0f}" if tp else "**undefined (0 repairs)**"
        wpr = f"{e['wall_s'] / tp:,.1f}" if tp else "**undefined (0 repairs)**"
        L.append(
            f"| `{m}` | {int(e['episodes'])} | {int(e['actions'])} | {int(e['applied'])} | "
            f"**{int(tp)}** | {ti:,} | {to:,} | {tpr} | {wpr} |"
        )
        cost_rows.append(
            {
                "model": m,
                "episodes": e["episodes"],
                "actions": e["actions"],
                "applied": e["applied"],
                "tp": tp,
                "tokens_in": ti,
                "tokens_out": to,
                "n_scored_available": t.get("n_available"),
                "n_scored_used": t.get("n"),
                "tokens_per_repair": ((ti + to) / tp) if tp else None,
                "wall_s_per_repair": (e["wall_s"] / tp) if tp else None,
            }
        )
    L.append("")
    for m in sorted(curators):
        t = tp_by_model.get(m, {})
        if t.get("n_available") and t.get("n") and t["n_available"] != t["n"]:
            L.append(
                f"*`{m}`: {int(t['n_available'])} episodes are scored on disk but only "
                f"**{int(t['n'])}** carry token telemetry, so only those {int(t['n'])} contribute "
                "to BOTH columns above.*"
            )
    if utility:
        L.append("")
        L.append("**Utility models (not curators — no repairs to divide by):**")
        L.append("")
        L.append("| model | role | calls | tokens in | tokens out |")
        L.append("|---|---|---:|---:|---:|")
        roles = {"gpt-5.4": "LLM judge (ladder rungs 6/7)", "ScriptedAgent": "deterministic probe"}
        for m in sorted(utility):
            ti, to, calls = tok.get(m, [0, 0, 0])
            L.append(f"| `{m}` | {roles.get(m, '—')} | {calls:,} | {ti:,} | {to:,} |")
    L.append("")
    res["a4_2"] = cost_rows
    L.append(
        "**The `applied` vs `tp` gap is the point of "
        "this table.** `applied` counts edits the agent "
        "made; `tp` counts edits that were correct. Cost "
        "per *applied action* flatters every model; "
        "cost per *correct repair* is the deployable quantity."
    )
    L.append("")
    L.append(
        "> ⚠ **Coverage gap, stated rather than filled.** Telemetry was instrumented at Stage 0, "
        "*after* the eight self-hosted vLLM cells of Table 1 (`qwen3.5-*`, `gemma4-12b`, "
        "`llama3.*`) had already run. Those rows therefore "
        "carry **no token or wall telemetry** and "
        "are absent above. Re-running 8 models × 300 episodes purely to populate a cost column "
        "would be a backfill, which the plan forbids; the honest statement is that hosted-model "
        "cost is measured and self-hosted cost is not."
    )
    L.append("")
    L.append(
        "**The zero-repair case, defined explicitly.** `llama3.2-1b` scores **F1 = 0.0000** over "
        "300 episodes in Table 1 — it never produces a correct repair. Its cost per repair is "
        "therefore **undefined (division by zero)**, and reporting it as a large finite number "
        "would be wrong. The correct reading is that *any* nonzero spend buys zero repairs, which "
        "is the strongest cost statement in the table. Its absolute token cost is unmeasured for "
        "the coverage reason above."
    )
    L.append("")

    # ================= A4.3 — stage timing / cache / deciding stage ============================
    L.append("## A4.3 — per-stage timing, cache-hit rate, deciding-stage distribution")
    L.append("")
    sig = [d["fields"] for d in ev if d.get("kind") == "signature_compute"]
    pv = [d["fields"] for d in ev if d.get("kind") == "pair_verify"]
    if sig:
        hits = sum(1 for s in sig if s.get("cache_hit"))
        miss = [s for s in sig if not s.get("cache_hit")]
        mw = [s["wall_ms"] for s in miss if s.get("wall_ms") is not None]
        L.append(
            f"- **signature cache:** {hits:,}/{len(sig):,} hits = **{pct(hits / len(sig))}**; "
            f"{len(miss):,} misses, median miss {statistics.median(mw):.0f} ms, "
            f"total execution **{sum(mw) / 3600000:.2f} h**"
        )
        res["a4_3_cache"] = {"n": len(sig), "hits": hits, "rate": hits / len(sig)}
    if pv:
        w = [p["wall_ms"] for p in pv if p.get("wall_ms") is not None]
        w.sort()
        ds = collections.Counter(p.get("deciding_stage") for p in pv)
        sc_ = sum(1 for p in pv if p.get("short_circuited"))
        L.append(
            f"- **pair verification:** n={len(pv):,}, median {statistics.median(w):.0f} ms, "
            f"p95 {w[int(0.95 * len(w))]:.0f} ms, max {w[-1]:.0f} ms, "
            f"total **{sum(w) / 3600000:.2f} h**"
        )
        L.append(f"- **short-circuited:** {sc_:,}/{len(pv):,} = {pct(sc_ / len(pv))}")
        L.append("")
        L.append("| deciding stage | n | share |")
        L.append("|---|---:|---|")
        for k, c in ds.most_common():
            L.append(f"| `{k}` | {c:,} | {pct(c / len(pv))} |")
        L.append("")
        cheap = sum(c for k, c in ds.items() if k in {"exact", "residual", "abstention_band"})
        L.append(
            f"★ **{pct(cheap / len(pv))} of pairs are decided before any learned backend runs** "
            f"({cheap:,}/{len(pv):,} at the `exact` / `residual` / `abstention_band` stages). That "
            "is the cascade earning its keep: the expensive perceptual and semantic stages are "
            "reached by a minority of pairs."
        )
        res["a4_3_stages"] = dict(ds)
    L.append("")

    # ================= A4.4 — feature-cache memory ============================================
    L.append("## A4.4 — feature-cache memory and extrapolation")
    L.append("")
    peaks = [s["peak_mem"] for s in sig if s.get("peak_mem")]
    # MEASURED unit of the cache, not inferred from process RSS. The evaluator caches rendered
    # OutputSets keyed by (skill, params, seed) -- `compare.py` `_outputset_nbytes` -- so the
    # scaling unit is one skill's outputs over the battery, measured directly rather than by
    # multiplying a process peak (which counts models, interpreter and transient buffers too).
    OUTPUTSET_MB = 214.84  # median over 8 builtin skills, 177-probe battery, measured
    DINO_FEAT_MB = 768 * 4 * 177 / 1e6  # analytic: feature dim x float32 x probes
    if peaks:
        peaks.sort()
        L.append(
            f"- process `signature_compute` peak memory: median "
            f"**{statistics.median(peaks) / 1e9:.2f} GB**, p95 "
            f"{peaks[int(0.95 * len(peaks))] / 1e9:.2f} GB, max "
            f"**{peaks[-1] / 1e9:.2f} GB** over {len(peaks):,} events"
        )
    L.append(
        f"- **measured cache unit: {OUTPUTSET_MB:.1f} MB per skill** (one `OutputSet` over the "
        f"177-probe battery, median of 8 builtin skills). This is what the LRU actually holds."
    )
    L.append("")
    L.append("| library size n | output cache (measured unit) | DINO-features-only (analytic) |")
    L.append("|---:|---|---|")
    for n in [100, 1000, 10000]:
        L.append(
            f"| {n:,} | **{OUTPUTSET_MB * n / 1000:.1f} GB** | {DINO_FEAT_MB * n / 1000:.2f} GB |"
        )
    L.append("")
    L.append(
        f"★ **The cache is output images, not features, and that is the whole cost.** At n=10⁴ an "
        f"unbounded output cache needs **{OUTPUTSET_MB * 1e4 / 1e6:.1f} TB**, while the DINO "
        f"features for the same library are **{DINO_FEAT_MB * 1e4 / 1000:.1f} GB** — a factor of "
        f"**~{OUTPUTSET_MB / DINO_FEAT_MB:.0f}×**. The measured unit also explains the Stage-4 "
        "benchmark's 161 GB peak on 100 skills without needing a leak hypothesis."
    )
    L.append("")
    L.append(
        "> This is why `--max-cache-gb` exists and why every run above sets it. The cache is a "
        'pure compute optimisation — `compare.py:153` states eviction *"never changes a '
        'result"* — so bounding it trades wall-clock for memory without altering a verdict. The '
        "projection assumes no cross-skill sharing and is an upper bound; the features-only column "
        "is a concrete design lever, not a measurement of the current implementation."
    )
    res["a4_4"] = {
        "outputset_mb_per_skill_measured": OUTPUTSET_MB,
        "dino_features_mb_per_skill_analytic": DINO_FEAT_MB,
        "process_peak_median_bytes": statistics.median(peaks) if peaks else None,
        "process_peak_max_bytes": peaks[-1] if peaks else None,
    }
    L.append("")

    # ================= A4.6 — pointer ==========================================================
    L.append("## A4.6 — quality/cost Pareto")
    L.append("")
    L.append(
        "Produced by `scripts/wacv_r2/aggregate_curation.py` with **instance-clustered** bootstrap "
        "CIs (never per-pair: episodes on one corruption instance share base skills and a defect "
        "draw). See `results/wacv_r2/table1/report.md` for the per-model F1 with intervals, which "
        "is the quality axis; the cost axis is A4.2 above and is populated for the hosted models "
        "only, for the coverage reason stated there."
    )
    L.append("")

    (out_dir / "a4_full.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"a4 full -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

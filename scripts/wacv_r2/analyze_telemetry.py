#!/usr/bin/env python
"""A4 — turn the telemetry JSONL into the cost/scaling report.

Reports tokens and latency, never dollars: prices change and a run has to stay reproducible, so
USD is a separate analysis-time multiplication against a timestamped price table.

Latency is reported as median AND p95. A mean hides the tail, and the tail is what a practitioner
feels — the pair-verification distribution here is strongly right-skewed because the subsumption
grid search runs only on some pairs.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Any


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(path: Path) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    pair_ms: list[float] = []
    deciding: Counter[str] = Counter()
    short_circuit = 0
    sig_hit = 0
    sig_miss_ms: list[float] = []
    llm: list[tuple[float, int, int]] = []
    llm_model: Counter[str] = Counter()
    episodes: list[dict[str, Any]] = []
    cand: list[dict[str, Any]] = []
    manifest: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        f = d.get("fields", {})
        kinds[d["kind"]] += 1
        if d["kind"] == "run_manifest":
            manifest = f
        elif d["kind"] == "pair_verify":
            pair_ms.append(f["wall_ms"])
            deciding[f["deciding_stage"]] += 1
            short_circuit += int(bool(f["short_circuited"]))
        elif d["kind"] == "signature_compute":
            if f["cache_hit"]:
                sig_hit += 1
            else:
                sig_miss_ms.append(f["wall_ms"])
        elif d["kind"] == "llm_call":
            llm.append((f["wall_ms"], f["tokens_in"], f["tokens_out"]))
            llm_model[f.get("model", "?")] += 1
        elif d["kind"] == "agent_episode":
            episodes.append(f)
        elif d["kind"] == "candidate_gen":
            cand.append(f)
    n_sig = sig_hit + len(sig_miss_ms)
    return {
        "file": str(path),
        "events": dict(kinds),
        "run": {
            k: manifest.get(k)
            for k in ("hostname", "gpu_name", "torch_version", "cuda_version", "git_sha")
        }
        if manifest
        else None,
        "candidate_generation": cand,
        "signature_cache": {
            "calls": n_sig,
            "hits": sig_hit,
            "misses": len(sig_miss_ms),
            "hit_rate": (sig_hit / n_sig) if n_sig else None,
            "miss_median_ms": st.median(sig_miss_ms) if sig_miss_ms else None,
            "miss_total_s": round(sum(sig_miss_ms) / 1000, 1),
        },
        "pair_verify": {
            "n": len(pair_ms),
            "median_ms": st.median(pair_ms) if pair_ms else None,
            "p95_ms": pct(pair_ms, 0.95),
            "max_ms": max(pair_ms) if pair_ms else None,
            "total_h": round(sum(pair_ms) / 3_600_000, 3),
            "deciding_stage": dict(deciding),
            "short_circuited": short_circuit,
            "short_circuit_rate": (short_circuit / len(pair_ms)) if pair_ms else None,
        },
        "llm": {
            "calls": len(llm),
            "models": dict(llm_model),
            "median_ms": st.median([x[0] for x in llm]) if llm else None,
            "p95_ms": pct([x[0] for x in llm], 0.95),
            "tokens_in_total": sum(x[1] for x in llm),
            "tokens_out_total": sum(x[2] for x in llm),
            "tokens_in_per_call_median": st.median([x[1] for x in llm]) if llm else None,
        },
        "episodes": {
            "n": len(episodes),
            "median_actions": st.median([e["n_actions"] for e in episodes]) if episodes else None,
            "median_wall_min": round(st.median([e["wall_ms"] for e in episodes]) / 60000, 2)
            if episodes
            else None,
            "status_totals": {
                k: sum(e[k] for e in episodes)
                for k in ("n_applied", "n_rejected", "n_blocked", "n_invalid")
            }
            if episodes
            else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/wacv_r2/telemetry_*.jsonl")
    ap.add_argument("--out", default="results/wacv_r2/a4_cost")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(".")
    files = sorted(root.glob(args.glob))
    summaries = [summarize(p) for p in files if p.stat().st_size > 0]

    L = [
        "# A4 — measured cost and scaling",
        "",
        "Tokens and latency only; USD is applied at "
        "analysis time from a timestamped price table so a run stays reproducible.",
        "",
    ]
    for s in summaries:
        L += [f"## `{Path(s['file']).name}`", ""]
        if s["run"]:
            L.append(
                "- host `{hostname}` · GPU `{gpu_name}` · torch `{torch_version}` / CUDA "
                "`{cuda_version}`".format(**{k: v or "?" for k, v in s["run"].items()})
            )
        L.append(f"- events: {s['events']}")
        for c in s["candidate_generation"]:
            L.append(
                f"- **candidate screening**: {c['n_candidates']} of {c['n_all_pairs']} all-pairs "
                f"kept → **{c['reduction']:.1%} reduction** at radius {c['radius']} "
                f"({c['wall_ms']:.0f} ms for n={c['n_skills']})"
            )
        sc = s["signature_cache"]
        if sc["calls"]:
            L.append(
                f"- **signature cache**: {sc['hits']}/{sc['calls']} hits = "
                f"**{sc['hit_rate']:.1%} hit rate**; {sc['misses']} misses at median "
                f"{sc['miss_median_ms']:.0f} ms, {sc['miss_total_s']} s total execution"
            )
        pv = s["pair_verify"]
        if pv["n"]:
            L.append(
                f"- **pair verification**: n={pv['n']}, median {pv['median_ms']:.0f} ms, "
                f"p95 {pv['p95_ms']:.0f} ms, max {pv['max_ms']:.0f} ms, {pv['total_h']} h total"
            )
            L.append(f"  - deciding stage: {pv['deciding_stage']}")
            L.append(
                f"  - settled without reaching a learned backend: {pv['short_circuited']} "
                f"({pv['short_circuit_rate']:.1%})"
            )
        lm = s["llm"]
        if lm["calls"]:
            L.append(
                f"- **LLM**: {lm['calls']} calls {lm['models']}, median {lm['median_ms']:.0f} ms, "
                f"p95 {lm['p95_ms']:.0f} ms; **{lm['tokens_in_total']:,} tokens in / "
                f"{lm['tokens_out_total']:,} out** (median {lm['tokens_in_per_call_median']:.0f} "
                "in per call)"
            )
        ep = s["episodes"]
        if ep["n"]:
            L.append(
                f"- **episodes**: {ep['n']}, median {ep['median_actions']:.0f} actions in "
                f"{ep['median_wall_min']:.1f} min; statuses {ep['status_totals']}"
            )
        L.append("")
    L += [
        "## Caveat on the deciding-stage distribution",
        "",
        "The classifier is hierarchical and stop-at-first, so the deciding stage shows how often "
        "the expensive learned stages are reached at all. It must NOT be converted into a "
        "forward-pass saving: the design is eager-per-stage with output caching — "
        "`BatteryEvaluator` caches outputs per `(skill, params, seed)` and the backends "
        "batch-extract features per stage — so short-circuiting saves *distance computation*, not "
        'the backbone passes, which are already paid. Claiming both "features are cached" and '
        '"short-circuiting avoids forward passes" would be self-contradictory.',
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / "summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"a4 cost -> {out_dir}/report.md ({len(summaries)} telemetry files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

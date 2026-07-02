"""Rendering the Phase-4 artifacts (CLAUDE.md §3.5.9, §5) — tables, CSVs, manifest, figure.

Everything a reviewer needs to read the go/no-go checkpoint: the per-track mergeable decision
and safety numbers, the output verifier's per-relation P/R/F1, and the **divergence table**
(output-vs-text disagreement by true relation, hard-negative slice separate). A run manifest
ties every number to its config/seeds/backends so nothing is a floating placeholder.

The figure is optional: if ``matplotlib`` is installed it is rendered; otherwise the underlying
table is still emitted (CSV + markdown), so the figure is one command away and no number is
ever invented to fill a plot.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from viscurate.benchmark.human_review import extract_review_slice, write_review_template
from viscurate.benchmark.metrics import (
    PRF,
    divergence_by_true_relation,
    mergeable_prf,
    precision_on_distinct,
)
from viscurate.benchmark.runner import BenchmarkResult
from viscurate.benchmark.tracks import Verdict
from viscurate.equivalence.relations import Relation
from viscurate.skills.model import SkillSpec

__all__ = [
    "render_markdown_report",
    "try_plot_divergence",
    "write_manifest",
    "write_report",
]


@dataclass(frozen=True)
class TextOperatingPoint:
    """A swept operating point for a score-bearing text judge."""

    text_track: str
    threshold: float
    precision: float
    recall: float
    f1: float
    false_merge_rate: float
    hard_negative_false_merge_rate: float
    divergence_all: float
    text_over_merge_all: int
    text_under_merge_all: int
    selected: bool = False


def _prf3(p: PRF) -> str:
    return f"{p.precision:.3f}/{p.recall:.3f}/{p.f1:.3f}"


def render_markdown_report(result: BenchmarkResult) -> str:
    """The human-readable divergence report (markdown)."""
    m = result.meta
    lines: list[str] = []
    lines.append("# VisCurate — Equivalence Benchmark (Phase 4)\n")
    lines.append(
        f"- skills: **{m.get('n_skills')}**  · scored pairs: **{m.get('n_pairs')}**  · "
        f"seed: `{m.get('seed')}`"
    )
    lines.append(
        f"- backends — perceptual: `{m.get('perceptual_backend')}`, "
        f"semantic: `{m.get('semantic_backend')}`, clip: `{m.get('clip_backend')}`"
    )
    calibrated = m.get("thresholds_calibrated")
    lines.append(
        f"- thresholds calibrated: **{calibrated}**"
        + ("" if calibrated else "  ⚠️ numbers are provisional until calibration on a labeled split")
    )
    lines.append("")

    # -- mergeable decision per track ------------------------------------------
    lines.append("## Mergeable decision per track (positive = EXACT ∪ PERCEPTUAL)\n")
    lines.append(
        "| track | kind | ran | P/R/F1 | false-merge (DISTINCT) | false-merge (hard-neg) | "
        "abstention |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for t in (result.output_track, *result.text_tracks):
        if not t.ran:
            lines.append(f"| {t.name} | {t.kind} | no | — | — | — | — _( {t.note} )_ |")
            continue
        prf = result.mergeable(t.name)
        safe_all = result.safety(t.name)
        safe_hard = result.safety(t.name, hard_negatives_only=True)
        abst = result.abstention(t.name)
        lines.append(
            f"| {t.name} | {t.kind} | yes | {_prf3(prf)} | "
            f"{safe_all.false_merges}/{safe_all.n_distinct} "
            f"({safe_all.false_merge_rate:.2f}) | "
            f"{safe_hard.false_merges}/{safe_hard.n_distinct} "
            f"({safe_hard.false_merge_rate:.2f}) | {abst:.2f} |"
        )
    lines.append("")

    # -- output verifier per-relation -----------------------------------------
    lines.append("## Output verifier — per-relation precision/recall/F1\n")
    lines.append("| relation | P/R/F1 | support |")
    lines.append("|---|---|---|")
    for rel, prf in result.per_relation(result.output_track.name).items():
        lines.append(f"| {rel.value} | {_prf3(prf)} | {prf.support} |")
    lines.append("")

    # -- divergence tables -----------------------------------------------------
    lines.append("## Divergence — output verifier vs text judges (mergeable decision)\n")
    lines.append(
        "_text over-merge_ = text says merge, output says no (the silent-merge danger); "
        "_text under-merge_ = output finds redundancy text misses.\n"
    )
    any_ran = False
    for t in result.text_tracks:
        if not t.ran:
            lines.append(f"### vs {t.name}\n\n_not run: {t.note}_\n")
            continue
        any_ran = True
        lines.append(f"### vs {t.name}\n")
        lines.append("| true relation | n | disagree | rate | text over-merge | text under-merge |")
        lines.append("|---|---|---|---|---|---|")
        for row in result.divergence(t.name):
            lines.append(
                f"| {row.slice_name} | {row.n} | {row.disagree} | {row.disagree_rate:.2f} | "
                f"{row.text_over_merge} | {row.text_under_merge} |"
            )
        lines.append("")
    if not any_ran:
        lines.append("_No text track ran._\n")

    ops = text_operating_points(result)
    selected_ops = [op for op in ops if op.selected]
    if selected_ops:
        lines.append("## Text-judge swept operating points\n")
        lines.append(
            "_Best F1_ sweeps the score threshold of each text judge at its own operating point, "
            "so the fixed default threshold is not the only baseline reported.\n"
        )
        lines.append(
            "| track | best threshold | P/R/F1 | false-merge (DISTINCT) | "
            "false-merge (hard-neg) | divergence (ALL) |"
        )
        lines.append("|---|---:|---|---:|---:|---:|")
        for op in selected_ops:
            lines.append(
                f"| {op.text_track} | {op.threshold:.4f} | "
                f"{op.precision:.3f}/{op.recall:.3f}/{op.f1:.3f} | "
                f"{op.false_merge_rate:.3f} | {op.hard_negative_false_merge_rate:.3f} | "
                f"{op.divergence_all:.3f} |"
            )
        lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- **κ (human agreement):** pending real annotation — the judgment-laden slice is "
        "exported to `review_template.json`. The κ value is computed from completed annotator "
        "files; it is never fabricated (CLAUDE.md §5)."
    )
    lines.append(
        "- The **answer key** is the designed graph `G0` (CLAUDE.md §2.5), fixed before any "
        "metric runs; SEMANTIC/SUBSUMPTION labels are design intent pending human re-certification."
    )
    return "\n".join(lines) + "\n"


def _write_divergence_csv(result: BenchmarkResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "text_track",
                "true_relation",
                "n",
                "disagree",
                "disagree_rate",
                "text_over_merge",
                "text_under_merge",
            ]
        )
        for t in result.text_tracks:
            if not t.ran:
                continue
            for row in result.divergence(t.name):
                w.writerow(
                    [
                        t.name,
                        row.slice_name,
                        row.n,
                        row.disagree,
                        f"{row.disagree_rate:.4f}",
                        row.text_over_merge,
                        row.text_under_merge,
                    ]
                )


def _write_per_pair_csv(result: BenchmarkResult, path: Path) -> None:
    track_names = [t.name for t in result.text_tracks if t.ran]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = [
            "a",
            "b",
            "true_relation",
            "is_hard_negative",
            "output_relation",
            "output_mergeable",
            "output_score",
            "lpips",
            "dino_p90",
        ]
        header += [f"{n}_mergeable" for n in track_names]
        header += [f"{n}_score" for n in track_names]
        w.writerow(header)
        for oc in result.outcomes:
            row = [
                oc.pair[0],
                oc.pair[1],
                oc.truth.relation.value,
                int(oc.truth.is_hard_negative),
                oc.output.relation.value,
                int(oc.output.mergeable),
                f"{oc.output.score:.5f}",
                f"{oc.measurement.lpips:.5f}",
                f"{oc.measurement.dino_p90:.5f}",
            ]
            row += [int(oc.text[n].mergeable) if n in oc.text else "" for n in track_names]
            row += [
                f"{oc.text[n].score:.5f}"
                if n in oc.text and math.isfinite(oc.text[n].score)
                else ""
                for n in track_names
            ]
            w.writerow(row)


def _thresholds_for_scores(scores: Sequence[float]) -> list[float]:
    finite = sorted({float(s) for s in scores if math.isfinite(float(s))})
    if not finite:
        return []
    below_all = max(0.0, min(finite) - 1.0e-9)
    above_all = min(1.0 + 1.0e-9, max(finite) + 1.0e-9)
    return sorted({below_all, *finite, above_all})


def text_operating_points(result: BenchmarkResult) -> list[TextOperatingPoint]:
    """Sweep score thresholds for text tracks and mark the best-F1 operating point."""
    rows: list[TextOperatingPoint] = []
    for track in result.text_tracks:
        if not track.ran:
            continue
        scored = {
            pair: verdict.score
            for pair, verdict in track.predictions.items()
            if math.isfinite(verdict.score)
        }
        if not scored:
            continue
        candidates: list[TextOperatingPoint] = []
        for threshold in _thresholds_for_scores(tuple(scored.values())):
            pred = {
                pair: Verdict(
                    relation=Relation.EXACT if score >= threshold else Relation.DISTINCT,
                    mergeable=score >= threshold,
                    score=score,
                )
                for pair, score in scored.items()
            }
            prf = mergeable_prf(result.truth, pred)
            safety = precision_on_distinct(result.truth, pred)
            hard = precision_on_distinct(result.truth, pred, hard_negatives_only=True)
            divergence = {
                row.slice_name: row
                for row in divergence_by_true_relation(
                    result.output_track.predictions, pred, result.truth
                )
            }
            all_row = divergence.get("ALL")
            candidates.append(
                TextOperatingPoint(
                    text_track=track.name,
                    threshold=threshold,
                    precision=prf.precision,
                    recall=prf.recall,
                    f1=prf.f1,
                    false_merge_rate=safety.false_merge_rate,
                    hard_negative_false_merge_rate=hard.false_merge_rate,
                    divergence_all=0.0 if all_row is None else all_row.disagree_rate,
                    text_over_merge_all=0 if all_row is None else all_row.text_over_merge,
                    text_under_merge_all=0 if all_row is None else all_row.text_under_merge,
                )
            )
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda op: (
                op.f1,
                op.precision,
                -op.false_merge_rate,
                -op.hard_negative_false_merge_rate,
                op.threshold,
            ),
        )
        rows.extend(
            TextOperatingPoint(
                text_track=op.text_track,
                threshold=op.threshold,
                precision=op.precision,
                recall=op.recall,
                f1=op.f1,
                false_merge_rate=op.false_merge_rate,
                hard_negative_false_merge_rate=op.hard_negative_false_merge_rate,
                divergence_all=op.divergence_all,
                text_over_merge_all=op.text_over_merge_all,
                text_under_merge_all=op.text_under_merge_all,
                selected=op is best,
            )
            for op in candidates
        )
    return rows


def _write_text_operating_points_csv(result: BenchmarkResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "text_track",
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_merge_rate",
                "hard_negative_false_merge_rate",
                "divergence_all",
                "text_over_merge_all",
                "text_under_merge_all",
                "selected",
            ]
        )
        for op in text_operating_points(result):
            w.writerow(
                [
                    op.text_track,
                    f"{op.threshold:.6f}",
                    f"{op.precision:.6f}",
                    f"{op.recall:.6f}",
                    f"{op.f1:.6f}",
                    f"{op.false_merge_rate:.6f}",
                    f"{op.hard_negative_false_merge_rate:.6f}",
                    f"{op.divergence_all:.6f}",
                    op.text_over_merge_all,
                    op.text_under_merge_all,
                    int(op.selected),
                ]
            )


def write_manifest(result: BenchmarkResult, path: Path, extra: Mapping[str, object]) -> None:
    """Record config/seeds/backends so every reported number traces back (CLAUDE.md §5, §9)."""
    manifest = {
        "phase": 4,
        "kind": "equivalence_benchmark",
        **dict(result.meta),
        "tracks": [
            {"name": t.name, "kind": t.kind, "ran": t.ran, "note": t.note}
            for t in (result.output_track, *result.text_tracks)
        ],
        **dict(extra),
    }
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def try_plot_divergence(result: BenchmarkResult, path: Path) -> bool:
    """Render a grouped bar chart of disagreement rate by true relation; False if no matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ran = [t for t in result.text_tracks if t.ran]
    if not ran:
        return False
    rows0 = result.divergence(ran[0].name)
    slices = [r.slice_name for r in rows0]
    x = range(len(slices))
    width = 0.8 / max(1, len(ran))
    fig, ax = plt.subplots(figsize=(max(6, len(slices) * 1.2), 4))
    for i, t in enumerate(ran):
        rates = [r.disagree_rate for r in result.divergence(t.name)]
        ax.bar([xi + i * width for xi in x], rates, width=width, label=t.name)
    ax.set_xticks([xi + width * (len(ran) - 1) / 2 for xi in x])
    ax.set_xticklabels(slices, rotation=30, ha="right")
    ax.set_ylabel("output-vs-text disagreement rate")
    ax.set_title("Divergence by true relation (Phase 4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def write_report(
    result: BenchmarkResult,
    out_dir: str | Path,
    *,
    specs: Sequence[SkillSpec],
    manifest_extra: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write the full Phase-4 artifact set to ``out_dir``; return the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["report"] = out / "report.md"
    paths["report"].write_text(render_markdown_report(result), encoding="utf-8")

    paths["divergence_csv"] = out / "divergence.csv"
    _write_divergence_csv(result, paths["divergence_csv"])

    paths["pairs_csv"] = out / "pairs.csv"
    _write_per_pair_csv(result, paths["pairs_csv"])

    paths["text_operating_points_csv"] = out / "text_operating_points.csv"
    _write_text_operating_points_csv(result, paths["text_operating_points_csv"])

    paths["manifest"] = out / "manifest.json"
    write_manifest(result, paths["manifest"], manifest_extra or {})

    spec_by_id = {s.id: s for s in specs}
    review = extract_review_slice(result, spec_by_id)
    paths["review_template"] = write_review_template(review, out / "review_template.json")

    if try_plot_divergence(result, out / "divergence.png"):
        paths["figure"] = out / "divergence.png"

    return paths

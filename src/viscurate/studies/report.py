"""Phase-8 report writer: aggregate tables, Pareto front, ablation, and manifest."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from viscurate.studies.metrics import (
    AblationResult,
    AggregateRow,
    StudyPoint,
    SummaryStats,
    aggregate_pareto_front,
    aggregate_points,
    construct_validity,
    vision_matters_ablation,
)

__all__ = [
    "render_markdown_report",
    "try_plot_pareto",
    "write_study_report",
]


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def _pm(stats: SummaryStats) -> str:
    if stats.n == 0:
        return "n/a"
    return f"{stats.mean:.3f} [{stats.ci95_low:.3f}, {stats.ci95_high:.3f}]"


def render_markdown_report(
    points: Sequence[StudyPoint],
    *,
    title: str = "VisCurate — Phase 8 Studies",
    output_gate: str = "output",
    text_gate: str = "text",
) -> str:
    """Render the human-readable Phase-8 report."""
    aggregates = aggregate_points(points)
    pareto = aggregate_pareto_front(aggregates)
    corr = construct_validity(points)
    ablation = vision_matters_ablation(points, output_gate=output_gate, text_gate=text_gate)
    methods = sorted({p.method for p in points})

    lines: list[str] = [f"# {title}\n"]
    lines.append(f"- seed-level points: **{len(points)}**")
    lines.append(f"- methods: {', '.join(f'`{m}`' for m in methods) if methods else '_none_'}")
    lines.append(
        "- warning: this report aggregates supplied run artifacts; it does not execute "
        "experiments or invent missing results."
    )
    lines.append("")

    lines.append("## Curation Pareto Inputs\n")
    lines.append(
        "| method | gate | rho | composition | mode | n | downstream success | compression | "
        "action cost | intrinsic | action F1 |"
    )
    lines.append("|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for row in aggregates:
        action_f1 = "n/a" if row.action_f1 is None else _pm(row.action_f1)
        lines.append(
            f"| {row.method} | {row.gate or '—'} | {row.rho:.2f} | {row.composition} | "
            f"{row.mode} | {row.n} | {_pm(row.success)} | {_pm(row.compression)} | "
            f"{_pm(row.action_cost)} | {_pm(row.intrinsic_score)} | {action_f1} |"
        )
    lines.append("")

    lines.append("## Pareto Front\n")
    lines.append(
        "_Non-dominated rows maximize downstream success and compression while minimizing "
        "action cost._\n"
    )
    lines.append("| method | gate | rho | composition | success | compression | action cost |")
    lines.append("|---|---|---:|---|---:|---:|---:|")
    for row in pareto:
        lines.append(
            f"| {row.method} | {row.gate or '—'} | {row.rho:.2f} | {row.composition} | "
            f"{row.success.mean:.3f} | {row.compression.mean:.3f} | {row.action_cost.mean:.3f} |"
        )
    if not pareto:
        lines.append("| _none_ | | | | | | |")
    lines.append("")

    lines.append("## Construct Validity\n")
    lines.append(
        "Intrinsic curation score is correlated with downstream success across seed-level "
        "libraries.\n"
    )
    lines.append("| n | Pearson | Spearman |")
    lines.append("|---:|---:|---:|")
    lines.append(f"| {corr.n} | {_fmt(corr.pearson)} | {_fmt(corr.spearman)} |")
    lines.append("")

    lines.append("## Vision-Matters Ablation\n")
    lines.append(
        f"Matched `{output_gate}`-gated minus `{text_gate}`-gated runs, by "
        "`(rho, composition, seed, mode)`.\n"
    )
    lines.append("| matched n | success delta | compression delta | action-cost delta |")
    lines.append("|---:|---:|---:|---:|")
    lines.append(
        f"| {len(ablation.deltas)} | {_pm(ablation.success_delta)} | "
        f"{_pm(ablation.compression_delta)} | {_pm(ablation.action_cost_delta)} |"
    )
    lines.append("")

    lines.append("## Notes\n")
    lines.append("- Reported intervals are normal-approximation 95% CIs over supplied seed rows.")
    lines.append(
        "- Action quality is scored against `ideal_actions.json`; `KEEP` entries are not required "
        "actions, and `retrieve`/`end` are not counted as predicted repairs."
    )
    return "\n".join(lines) + "\n"


def _write_points_csv(points: Sequence[StudyPoint], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "method",
                "gate",
                "rho",
                "composition",
                "seed",
                "mode",
                "downstream_success",
                "compression",
                "action_cost",
                "intrinsic_score",
                "action_precision",
                "action_recall",
                "action_f1",
                "mergeable_f1",
                "false_merge_rate",
                "metadata",
            ]
        )
        for p in points:
            w.writerow(
                [
                    p.method,
                    p.gate,
                    f"{p.rho:.6f}",
                    p.composition,
                    p.seed,
                    p.mode,
                    f"{p.downstream_success:.6f}",
                    p.compression,
                    p.action_cost,
                    f"{p.intrinsic_score:.6f}",
                    _fmt(p.action_precision),
                    _fmt(p.action_recall),
                    _fmt(p.action_f1),
                    _fmt(p.mergeable_f1),
                    _fmt(p.false_merge_rate),
                    json.dumps(p.metadata, sort_keys=True),
                ]
            )


def _write_aggregates_csv(rows: Sequence[AggregateRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "method",
                "gate",
                "rho",
                "composition",
                "mode",
                "n",
                "success_mean",
                "success_ci95_low",
                "success_ci95_high",
                "compression_mean",
                "action_cost_mean",
                "intrinsic_mean",
                "action_f1_mean",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.method,
                    r.gate,
                    f"{r.rho:.6f}",
                    r.composition,
                    r.mode,
                    r.n,
                    f"{r.success.mean:.6f}",
                    f"{r.success.ci95_low:.6f}",
                    f"{r.success.ci95_high:.6f}",
                    f"{r.compression.mean:.6f}",
                    f"{r.action_cost.mean:.6f}",
                    f"{r.intrinsic_score.mean:.6f}",
                    "" if r.action_f1 is None else f"{r.action_f1.mean:.6f}",
                ]
            )


def _write_ablation_csv(result: AblationResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rho",
                "composition",
                "seed",
                "mode",
                "output_success",
                "text_success",
                "success_delta",
                "output_compression",
                "text_compression",
                "compression_delta",
                "output_action_cost",
                "text_action_cost",
                "action_cost_delta",
            ]
        )
        for d in result.deltas:
            w.writerow(
                [
                    f"{d.rho:.6f}",
                    d.composition,
                    d.seed,
                    d.mode,
                    f"{d.output_success:.6f}",
                    f"{d.text_success:.6f}",
                    f"{d.success_delta:.6f}",
                    f"{d.output_compression:.6f}",
                    f"{d.text_compression:.6f}",
                    f"{d.compression_delta:.6f}",
                    f"{d.output_action_cost:.6f}",
                    f"{d.text_action_cost:.6f}",
                    f"{d.action_cost_delta:.6f}",
                ]
            )


def try_plot_pareto(rows: Sequence[AggregateRow], path: Path) -> bool:
    """Render success-vs-action-cost scatter with compression as marker size."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = sorted({r.method for r in rows})
    for method in methods:
        subset = [r for r in rows if r.method == method]
        sizes = [max(20.0, 30.0 + 20.0 * r.compression.mean) for r in subset]
        ax.scatter(
            [r.action_cost.mean for r in subset],
            [r.success.mean for r in subset],
            s=sizes,
            alpha=0.75,
            label=method,
        )
    ax.set_xlabel("action cost (mean)")
    ax.set_ylabel("downstream success (mean)")
    ax.set_title("Phase-8 curation Pareto inputs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def write_study_report(
    points: Sequence[StudyPoint],
    out_dir: str | Path,
    *,
    title: str = "VisCurate — Phase 8 Studies",
    output_gate: str = "output",
    text_gate: str = "text",
    manifest_extra: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write Phase-8 artifacts and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    aggregates = aggregate_points(points)
    pareto = aggregate_pareto_front(aggregates)
    corr = construct_validity(points)
    ablation = vision_matters_ablation(points, output_gate=output_gate, text_gate=text_gate)

    paths["report"] = out / "report.md"
    paths["report"].write_text(
        render_markdown_report(points, title=title, output_gate=output_gate, text_gate=text_gate),
        encoding="utf-8",
    )

    paths["points_csv"] = out / "points.csv"
    _write_points_csv(points, paths["points_csv"])

    paths["aggregates_csv"] = out / "aggregates.csv"
    _write_aggregates_csv(aggregates, paths["aggregates_csv"])

    paths["pareto_csv"] = out / "pareto.csv"
    _write_aggregates_csv(pareto, paths["pareto_csv"])

    paths["ablation_csv"] = out / "vision_matters_ablation.csv"
    _write_ablation_csv(ablation, paths["ablation_csv"])

    paths["correlations"] = out / "construct_validity.json"
    paths["correlations"].write_text(corr.model_dump_json(indent=2), encoding="utf-8")

    paths["ablation_summary"] = out / "vision_matters_ablation.json"
    paths["ablation_summary"].write_text(ablation.model_dump_json(indent=2), encoding="utf-8")

    paths["manifest"] = out / "manifest.json"
    manifest = {
        "phase": 8,
        "kind": "studies",
        "n_points": len(points),
        "n_aggregates": len(aggregates),
        "n_pareto": len(pareto),
        "methods": sorted({p.method for p in points}),
        "gates": sorted({p.gate for p in points if p.gate}),
        "construct_validity": corr.model_dump(mode="json"),
        "vision_matters": {
            "output_gate": output_gate,
            "text_gate": text_gate,
            "n_matched": len(ablation.deltas),
            "success_delta": ablation.success_delta.model_dump(mode="json"),
        },
        **dict(manifest_extra or {}),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    if try_plot_pareto(aggregates, out / "pareto.png"):
        paths["figure"] = out / "pareto.png"

    return paths

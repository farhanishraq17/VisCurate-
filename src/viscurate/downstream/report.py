"""Write Phase-7 downstream evaluation artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

from viscurate.downstream.evaluate import DownstreamResult

__all__ = ["render_markdown_report", "write_downstream_report"]


def render_markdown_report(result: DownstreamResult) -> str:
    """Human-readable downstream summary."""
    lines: list[str] = []
    lines.append("# VisCurate — Downstream Evaluation (Phase 7)\n")
    lines.append(f"- queries: **{result.n}**")
    lines.append(f"- solver: `{result.meta.get('solver')}`")
    lines.append(f"- thresholds calibrated: **{result.meta.get('thresholds_calibrated')}**")
    if not result.meta.get("thresholds_calibrated"):
        lines.append("- note: threshold values are wiring placeholders until Phase-4 calibration")
    lines.append("")
    lines.append("## Success\n")
    lines.append(f"- overall: **{result.success_rate():.3f}**")
    for split in sorted({s.split for s in result.scores}):
        n_split = sum(1 for s in result.scores if s.split == split)
        lines.append(f"- {split}: **{result.success_rate(split):.3f}** over {n_split} queries")
    lines.append("")
    lines.append("## Notes\n")
    lines.append(
        "- Success requires both reference-output match and task predicates; this report is a real "
        "run artifact, not a placeholder for Phase-8 study numbers."
    )
    return "\n".join(lines) + "\n"


def _write_scores_csv(result: DownstreamResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "query_id",
                "split",
                "success",
                "reference_match",
                "predicates_passed",
                "l_inf",
                "lpips",
                "expected_skill_ids",
                "used_skill_ids",
                "error",
            ]
        )
        for s in result.scores:
            w.writerow(
                [
                    s.query_id,
                    s.split,
                    int(s.success),
                    int(s.reference_match),
                    int(s.predicates_passed),
                    "" if s.l_inf is None else f"{s.l_inf:.6f}",
                    "" if s.lpips is None else f"{s.lpips:.6f}",
                    " ".join(s.expected_skill_ids),
                    " ".join(s.used_skill_ids),
                    s.error,
                ]
            )


def write_downstream_report(
    result: DownstreamResult,
    out_dir: str | Path,
    *,
    manifest_extra: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write report artifacts and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["report"] = out / "report.md"
    paths["report"].write_text(render_markdown_report(result), encoding="utf-8")

    paths["scores_csv"] = out / "scores.csv"
    _write_scores_csv(result, paths["scores_csv"])

    paths["scores_json"] = out / "scores.json"
    paths["scores_json"].write_text(result.model_dump_json(indent=2), encoding="utf-8")

    paths["summary"] = out / "summary.json"
    paths["summary"].write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")

    paths["manifest"] = out / "manifest.json"
    manifest = {**result.meta, **dict(manifest_extra or {})}
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return paths

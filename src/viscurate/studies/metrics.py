"""Phase-8 study metrics: curation quality, Pareto fronts, CIs, and ablations.

The lower phases produce real run artifacts: equivalence benchmark results, curation action logs,
corruption ideal-action keys, and downstream query scores. This module is the pure aggregation
layer over those artifacts. It never invents a number; callers either pass real result objects or
load rows from a manifest/CSV and the functions compute deterministic summaries.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from viscurate.benchmark.runner import BenchmarkResult
from viscurate.corruption.types import IdealAction, IdealActionKind
from viscurate.curation.actions import ActionKind, ActionResult, ActionStatus
from viscurate.curation.environment import EpisodeResult
from viscurate.downstream.evaluate import DownstreamResult

__all__ = [
    "AblationDelta",
    "AblationResult",
    "ActionScore",
    "AggregateRow",
    "CorrelationResult",
    "EquivalenceTrackSummary",
    "StudyPoint",
    "SummaryStats",
    "action_cost",
    "aggregate_pareto_front",
    "aggregate_points",
    "construct_validity",
    "equivalence_track_summaries",
    "intrinsic_curation_score",
    "load_study_points",
    "pareto_front",
    "score_actions",
    "study_point_from_results",
    "summarize",
    "vision_matters_ablation",
]

_MERGE_ACTIONS = {IdealActionKind.MERGE.value, IdealActionKind.PARAMETERIZE.value}
_SCORABLE_ACTIONS = {
    IdealActionKind.MERGE.value,
    IdealActionKind.PARAMETERIZE.value,
    IdealActionKind.MODIFY.value,
    IdealActionKind.REMOVE.value,
}


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SummaryStats(_Frozen):
    """Mean ± normal-approximation 95% CI over seed-level measurements."""

    n: int
    mean: float
    std: float
    ci95_low: float
    ci95_high: float


class ActionScore(_Frozen):
    """Action-log precision/recall/F1 against the corruption ideal-action key."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    n_ideal: int
    n_predicted: int


class StudyPoint(_Frozen):
    """One seed-level Phase-8 observation for a method on one ``L_ρ`` instance."""

    method: str
    gate: str = ""  # e.g. "output", "text", "oracle", "none"
    rho: float = Field(ge=0.0, le=1.0)
    composition: str
    seed: int
    mode: str = "single"
    downstream_success: float = Field(ge=0.0, le=1.0)
    compression: int = 0
    action_cost: int = Field(default=0, ge=0)
    intrinsic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    action_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    action_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    action_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    mergeable_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    false_merge_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("method", "composition", "mode")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("field must be non-empty")
        return value

    @property
    def group_key(self) -> tuple[str, str, float, str, str]:
        return (self.method, self.gate, self.rho, self.composition, self.mode)

    @property
    def ablation_key(self) -> tuple[float, str, int, str]:
        return (self.rho, self.composition, self.seed, self.mode)


class AggregateRow(_Frozen):
    """A method/configuration aggregate, usually across seeds."""

    method: str
    gate: str
    rho: float
    composition: str
    mode: str
    n: int
    success: SummaryStats
    compression: SummaryStats
    action_cost: SummaryStats
    intrinsic_score: SummaryStats
    action_f1: SummaryStats | None = None


class CorrelationResult(_Frozen):
    """Construct-validity correlation between intrinsic score and downstream success."""

    n: int
    pearson: float | None
    spearman: float | None


class AblationDelta(_Frozen):
    """Matched output-gated minus text-gated curation result for one seed/config."""

    rho: float
    composition: str
    seed: int
    mode: str
    output_success: float
    text_success: float
    success_delta: float
    output_compression: float
    text_compression: float
    compression_delta: float
    output_action_cost: float
    text_action_cost: float
    action_cost_delta: float


class AblationResult(_Frozen):
    """The Phase-8 vision-matters ablation summary."""

    output_gate: str
    text_gate: str
    deltas: tuple[AblationDelta, ...]
    success_delta: SummaryStats
    compression_delta: SummaryStats
    action_cost_delta: SummaryStats


class EquivalenceTrackSummary(_Frozen):
    """Study-1 row for one equivalence judge track."""

    track: str
    kind: str
    ran: bool
    mergeable_precision: float | None
    mergeable_recall: float | None
    mergeable_f1: float | None
    false_merge_rate: float | None
    hard_negative_false_merge_rate: float | None
    abstention_rate: float | None
    note: str = ""


def _bootstrap_interval(values: Sequence[float], *, samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(max(1, samples)):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def summarize(
    values: Sequence[float],
    *,
    ci_method: str = "normal",
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> SummaryStats:
    """Return mean/std/95% CI; supports normal approximation or deterministic bootstrap."""
    finite = [float(v) for v in values if math.isfinite(float(v))]
    n = len(finite)
    if n == 0:
        return SummaryStats(n=0, mean=0.0, std=0.0, ci95_low=0.0, ci95_high=0.0)
    mean = sum(finite) / n
    if n == 1:
        std = 0.0
        half = 0.0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in finite) / (n - 1))
        if ci_method == "normal":
            half = 1.96 * std / math.sqrt(n)
            return SummaryStats(
                n=n, mean=mean, std=std, ci95_low=mean - half, ci95_high=mean + half
            )
        if ci_method == "bootstrap":
            lo, hi = _bootstrap_interval(finite, samples=bootstrap_samples, seed=seed)
            return SummaryStats(n=n, mean=mean, std=std, ci95_low=lo, ci95_high=hi)
        raise ValueError(f"unknown ci_method: {ci_method!r}")
    return SummaryStats(n=n, mean=mean, std=std, ci95_low=mean - half, ci95_high=mean + half)


def _ideal_key(action: IdealAction) -> tuple[str, str, str] | None:
    kind = action.kind.value
    if kind == IdealActionKind.KEEP.value:
        return None
    secondary = action.secondary if kind in _MERGE_ACTIONS else ""
    return (kind, action.primary, secondary)


def _result_key(result: ActionResult) -> tuple[str, str, str] | None:
    action = result.action
    kind = action.kind.value
    if kind not in _SCORABLE_ACTIONS:
        return None
    secondary = action.secondary if kind in _MERGE_ACTIONS else ""
    return (kind, action.primary, secondary)


def score_actions(
    log: Sequence[ActionResult],
    ideal_actions: Sequence[IdealAction],
    *,
    applied_only: bool = True,
) -> ActionScore:
    """Score curation actions against the ideal-action key.

    ``merge`` and ``parameterize`` match on ``(kind, primary, secondary)`` because direction
    matters. ``modify`` and ``remove`` match on ``(kind, primary)``. ``KEEP`` ideals are not
    required actions, and ``retrieve``/``end`` do not count as predictions.
    """
    ideal = [_ideal_key(a) for a in ideal_actions]
    remaining: dict[tuple[str, str, str], int] = {}
    for key in ideal:
        if key is not None:
            remaining[key] = remaining.get(key, 0) + 1

    predicted: list[tuple[str, str, str]] = []
    for result in log:
        if applied_only and result.status is not ActionStatus.APPLIED:
            continue
        key = _result_key(result)
        if key is not None:
            predicted.append(key)

    tp = fp = 0
    for key in predicted:
        if remaining.get(key, 0) > 0:
            tp += 1
            remaining[key] -= 1
        else:
            fp += 1
    fn = sum(remaining.values())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ActionScore(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        n_ideal=tp + fn,
        n_predicted=len(predicted),
    )


def action_cost(log: Sequence[ActionResult]) -> int:
    """Count non-``end`` actions; this is the Pareto action-cost axis."""
    return sum(1 for r in log if r.action.kind is not ActionKind.END)


def intrinsic_curation_score(log: Sequence[ActionResult], score: ActionScore) -> float:
    """Intrinsic curation quality used for construct-validity checks.

    The core signal is ideal-action F1. Rejected/blocked/invalid actions are penalized because
    they spend budget without improving the library; this keeps the score independent of
    downstream query success.
    """
    cost = action_cost(log)
    if cost == 0:
        return 0.0
    nonproductive = sum(
        1
        for r in log
        if r.action.kind is not ActionKind.END
        and r.status in {ActionStatus.REJECTED, ActionStatus.BLOCKED, ActionStatus.INVALID}
    )
    penalty = max(0.0, 1.0 - 0.5 * (nonproductive / cost))
    return max(0.0, min(1.0, score.f1 * penalty))


def study_point_from_results(
    method: str,
    downstream: DownstreamResult,
    *,
    rho: float,
    composition: str,
    seed: int,
    mode: str = "single",
    gate: str = "",
    episode: EpisodeResult | None = None,
    ideal_actions: Sequence[IdealAction] = (),
    split: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> StudyPoint:
    """Build a seed-level study row from real Phase-6/7 result objects."""
    log = episode.log if episode is not None else ()
    scored = score_actions(log, ideal_actions) if episode is not None else None
    intrinsic = intrinsic_curation_score(log, scored) if scored is not None else 0.0
    return StudyPoint(
        method=method,
        gate=gate,
        rho=rho,
        composition=composition,
        seed=seed,
        mode=mode,
        downstream_success=downstream.success_rate(split),
        compression=episode.compression if episode is not None else 0,
        action_cost=action_cost(log),
        intrinsic_score=intrinsic,
        action_precision=None if scored is None else scored.precision,
        action_recall=None if scored is None else scored.recall,
        action_f1=None if scored is None else scored.f1,
        metadata=dict(metadata or {}),
    )


def aggregate_points(
    points: Sequence[StudyPoint],
    *,
    ci_method: str = "normal",
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> list[AggregateRow]:
    """Aggregate study points by ``(method, gate, rho, composition, mode)``."""
    groups: dict[tuple[str, str, float, str, str], list[StudyPoint]] = {}
    for point in points:
        groups.setdefault(point.group_key, []).append(point)

    rows: list[AggregateRow] = []
    for i, ((method, gate, rho, composition, mode), group) in enumerate(sorted(groups.items())):
        action_f1_values = [p.action_f1 for p in group if p.action_f1 is not None]
        group_seed = seed + i * 17
        rows.append(
            AggregateRow(
                method=method,
                gate=gate,
                rho=rho,
                composition=composition,
                mode=mode,
                n=len(group),
                success=summarize(
                    [p.downstream_success for p in group],
                    ci_method=ci_method,
                    bootstrap_samples=bootstrap_samples,
                    seed=group_seed,
                ),
                compression=summarize(
                    [float(p.compression) for p in group],
                    ci_method=ci_method,
                    bootstrap_samples=bootstrap_samples,
                    seed=group_seed,
                ),
                action_cost=summarize(
                    [float(p.action_cost) for p in group],
                    ci_method=ci_method,
                    bootstrap_samples=bootstrap_samples,
                    seed=group_seed,
                ),
                intrinsic_score=summarize(
                    [p.intrinsic_score for p in group],
                    ci_method=ci_method,
                    bootstrap_samples=bootstrap_samples,
                    seed=group_seed,
                ),
                action_f1=summarize(
                    action_f1_values,
                    ci_method=ci_method,
                    bootstrap_samples=bootstrap_samples,
                    seed=group_seed,
                )
                if action_f1_values
                else None,
            )
        )
    return rows


def _dominates(a: StudyPoint, b: StudyPoint) -> bool:
    ge = a.downstream_success >= b.downstream_success and a.compression >= b.compression
    le = a.action_cost <= b.action_cost
    strict = (
        a.downstream_success > b.downstream_success
        or a.compression > b.compression
        or a.action_cost < b.action_cost
    )
    return ge and le and strict


def pareto_front(points: Sequence[StudyPoint]) -> list[StudyPoint]:
    """Return non-dominated seed-level points (success/compression up, action cost down)."""
    front: list[StudyPoint] = []
    for point in points:
        if not any(_dominates(other, point) for other in points):
            front.append(point)
    return sorted(front, key=lambda p: (p.rho, p.composition, p.action_cost, -p.downstream_success))


def _agg_dominates(a: AggregateRow, b: AggregateRow) -> bool:
    ge = a.success.mean >= b.success.mean and a.compression.mean >= b.compression.mean
    le = a.action_cost.mean <= b.action_cost.mean
    strict = (
        a.success.mean > b.success.mean
        or a.compression.mean > b.compression.mean
        or a.action_cost.mean < b.action_cost.mean
    )
    return ge and le and strict


def aggregate_pareto_front(rows: Sequence[AggregateRow]) -> list[AggregateRow]:
    """Return non-dominated aggregate rows."""
    front: list[AggregateRow] = []
    for row in rows:
        if not any(_agg_dominates(other, row) for other in rows):
            front.append(row)
    return sorted(front, key=lambda r: (r.rho, r.composition, r.action_cost.mean, -r.success.mean))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(vx * vy)


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def construct_validity(points: Sequence[StudyPoint]) -> CorrelationResult:
    """Correlate intrinsic curation quality with downstream success across libraries."""
    rows = [p for p in points if math.isfinite(p.intrinsic_score)]
    xs = [p.intrinsic_score for p in rows]
    ys = [p.downstream_success for p in rows]
    pearson = _pearson(xs, ys)
    spearman = _pearson(_ranks(xs), _ranks(ys)) if len(xs) >= 2 else None
    return CorrelationResult(n=len(rows), pearson=pearson, spearman=spearman)


def _mean_point(points: Sequence[StudyPoint]) -> tuple[float, float, float]:
    n = len(points)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (
        sum(p.downstream_success for p in points) / n,
        sum(float(p.compression) for p in points) / n,
        sum(float(p.action_cost) for p in points) / n,
    )


def vision_matters_ablation(
    points: Sequence[StudyPoint],
    *,
    output_gate: str = "output",
    text_gate: str = "text",
    ci_method: str = "normal",
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> AblationResult:
    """Compare matched output-gated and text-gated curation runs."""
    output: dict[tuple[float, str, int, str], list[StudyPoint]] = {}
    text: dict[tuple[float, str, int, str], list[StudyPoint]] = {}
    for point in points:
        if point.gate == output_gate:
            output.setdefault(point.ablation_key, []).append(point)
        elif point.gate == text_gate:
            text.setdefault(point.ablation_key, []).append(point)

    deltas: list[AblationDelta] = []
    for key in sorted(set(output) & set(text)):
        rho, composition, seed, mode = key
        out_success, out_compression, out_cost = _mean_point(output[key])
        text_success, text_compression, text_cost = _mean_point(text[key])
        deltas.append(
            AblationDelta(
                rho=rho,
                composition=composition,
                seed=seed,
                mode=mode,
                output_success=out_success,
                text_success=text_success,
                success_delta=out_success - text_success,
                output_compression=out_compression,
                text_compression=text_compression,
                compression_delta=out_compression - text_compression,
                output_action_cost=out_cost,
                text_action_cost=text_cost,
                action_cost_delta=out_cost - text_cost,
            )
        )
    return AblationResult(
        output_gate=output_gate,
        text_gate=text_gate,
        deltas=tuple(deltas),
        success_delta=summarize(
            [d.success_delta for d in deltas],
            ci_method=ci_method,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        compression_delta=summarize(
            [d.compression_delta for d in deltas],
            ci_method=ci_method,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        action_cost_delta=summarize(
            [d.action_cost_delta for d in deltas],
            ci_method=ci_method,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    )


def equivalence_track_summaries(result: BenchmarkResult) -> list[EquivalenceTrackSummary]:
    """Build Study-1 rows from a Phase-4 benchmark result."""
    rows: list[EquivalenceTrackSummary] = []
    for track in (result.output_track, *result.text_tracks):
        if not track.ran:
            rows.append(
                EquivalenceTrackSummary(
                    track=track.name,
                    kind=track.kind,
                    ran=False,
                    mergeable_precision=None,
                    mergeable_recall=None,
                    mergeable_f1=None,
                    false_merge_rate=None,
                    hard_negative_false_merge_rate=None,
                    abstention_rate=None,
                    note=track.note,
                )
            )
            continue
        prf = result.mergeable(track.name)
        safety = result.safety(track.name)
        hard = result.safety(track.name, hard_negatives_only=True)
        rows.append(
            EquivalenceTrackSummary(
                track=track.name,
                kind=track.kind,
                ran=True,
                mergeable_precision=prf.precision,
                mergeable_recall=prf.recall,
                mergeable_f1=prf.f1,
                false_merge_rate=safety.false_merge_rate,
                hard_negative_false_merge_rate=hard.false_merge_rate,
                abstention_rate=result.abstention(track.name),
                note=track.note,
            )
        )
    return rows


def _coerce_float(value: object, *, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"expected scalar float-like value, got {type(value).__name__}")


def _coerce_optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"expected scalar float-like value, got {type(value).__name__}")


def _point_from_csv(row: Mapping[str, str]) -> StudyPoint:
    metadata: dict[str, Any] = {}
    if row.get("metadata"):
        parsed = json.loads(row["metadata"])
        if isinstance(parsed, dict):
            metadata = parsed
    return StudyPoint(
        method=row["method"],
        gate=row.get("gate", ""),
        rho=_coerce_float(row.get("rho")),
        composition=row["composition"],
        seed=int(_coerce_float(row.get("seed"))),
        mode=row.get("mode", "single") or "single",
        downstream_success=_coerce_float(row.get("downstream_success")),
        compression=int(_coerce_float(row.get("compression"))),
        action_cost=int(_coerce_float(row.get("action_cost"))),
        intrinsic_score=_coerce_float(row.get("intrinsic_score")),
        action_precision=_coerce_optional_float(row.get("action_precision")),
        action_recall=_coerce_optional_float(row.get("action_recall")),
        action_f1=_coerce_optional_float(row.get("action_f1")),
        mergeable_f1=_coerce_optional_float(row.get("mergeable_f1")),
        false_merge_rate=_coerce_optional_float(row.get("false_merge_rate")),
        metadata=metadata,
    )


def load_study_points(path: str | Path) -> list[StudyPoint]:
    """Load seed-level study rows from JSON or CSV.

    JSON may be either ``[point, ...]`` or ``{"points": [point, ...]}``. CSV uses the
    :class:`StudyPoint` field names as headers.
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with p.open("r", newline="", encoding="utf-8") as fh:
            return [_point_from_csv(row) for row in csv.DictReader(fh)]

    raw = json.loads(p.read_text(encoding="utf-8"))
    data: object = raw.get("points", raw) if isinstance(raw, dict) else raw
    if not isinstance(data, list):
        raise TypeError("study point JSON must be a list or an object with a 'points' list")
    return [StudyPoint.model_validate(item) for item in data]

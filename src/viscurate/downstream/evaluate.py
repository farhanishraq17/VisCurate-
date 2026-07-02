"""Run downstream query evaluation over a clean, corrupted, or curated library."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from viscurate.config import ThresholdConfig
from viscurate.downstream.build import load_array
from viscurate.downstream.predicates import PredicateResult, evaluate_predicates
from viscurate.downstream.query import QueryManifest
from viscurate.downstream.solver import SolverAgent, execute_plan
from viscurate.equivalence.backends import PerceptualBackend
from viscurate.probes.build import array_sha256
from viscurate.rng import SeedManager
from viscurate.skills.canonicalize import canonicalize, max_abs_pixel_diff
from viscurate.skills.registry import SkillRegistry

__all__ = ["DownstreamResult", "QueryScore", "run_downstream"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QueryScore(_Frozen):
    """The scored outcome for one query."""

    query_id: str
    split: str
    success: bool
    reference_match: bool
    predicates_passed: bool
    l_inf: float | None = None
    lpips: float | None = None
    error: str = ""
    solver_note: str = ""
    expected_skill_ids: tuple[str, ...] = ()
    used_skill_ids: tuple[str, ...] = ()
    predicate_results: tuple[PredicateResult, ...] = ()


class DownstreamResult(_Frozen):
    """A complete downstream run, ready to serialize into Phase-8 study artifacts."""

    scores: tuple[QueryScore, ...]
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.scores)

    def success_rate(self, split: str | None = None) -> float:
        rows = [s for s in self.scores if split is None or s.split == split]
        if not rows:
            return 0.0
        return sum(1 for s in rows if s.success) / len(rows)

    def summary(self) -> dict[str, object]:
        splits = sorted({s.split for s in self.scores})
        return {
            "n": self.n,
            "success_rate": self.success_rate(),
            "by_split": {
                split: {
                    "n": sum(1 for s in self.scores if s.split == split),
                    "success_rate": self.success_rate(split),
                }
                for split in splits
            },
        }


def _reference_match(
    *,
    l_inf: float,
    lpips: float | None,
    thresholds: ThresholdConfig,
) -> bool:
    if l_inf <= thresholds.exact_epsilon:
        return True
    return lpips is not None and lpips <= thresholds.perceptual_lpips


def run_downstream(
    manifest: QueryManifest,
    query_dir: str | Path,
    registry: SkillRegistry,
    solver: SolverAgent,
    *,
    thresholds: ThresholdConfig,
    perceptual: PerceptualBackend | None = None,
    seed: int = 0,
    splits: Iterable[str] | None = None,
    allow_untrusted: bool = False,
    meta: dict[str, Any] | None = None,
) -> DownstreamResult:
    """Evaluate ``solver`` against ``manifest`` using ``registry`` as the available library."""
    sm = SeedManager(seed)
    scores: list[QueryScore] = []
    selected = manifest.by_split(splits)
    for q in selected:
        input_image = load_array(query_dir, q.input_path)
        reference = load_array(query_dir, q.reference_path)
        if array_sha256(input_image) != q.input_sha256:
            raise ValueError(f"query input hash mismatch: {q.query_id}")
        if array_sha256(reference) != q.reference_sha256:
            raise ValueError(f"query reference hash mismatch: {q.query_id}")

        plan = solver.solve(q, registry)
        execution = execute_plan(
            registry,
            plan,
            input_image,
            seed=sm.child_seed("solve", q.query_id) & 0xFFFF_FFFF,
            allow_untrusted=allow_untrusted,
        )
        if not execution.ok or execution.output is None:
            scores.append(
                QueryScore(
                    query_id=q.query_id,
                    split=q.split,
                    success=False,
                    reference_match=False,
                    predicates_passed=False,
                    error=execution.error,
                    solver_note=plan.note,
                    expected_skill_ids=q.expected_skill_ids,
                    used_skill_ids=execution.used_skill_ids,
                )
            )
            continue

        out_c = canonicalize(execution.output)
        ref_c = canonicalize(reference)
        l_inf = max_abs_pixel_diff(out_c, ref_c)
        lpips = (
            perceptual.distance(out_c.rgb, ref_c.rgb)
            if perceptual is not None and out_c.rgb.shape == ref_c.rgb.shape
            else None
        )
        reference_match = _reference_match(l_inf=l_inf, lpips=lpips, thresholds=thresholds)
        predicate_results = evaluate_predicates(
            q.predicates,
            output=execution.output,
            reference=reference,
            input_image=input_image,
        )
        predicates_passed = all(r.passed for r in predicate_results)
        scores.append(
            QueryScore(
                query_id=q.query_id,
                split=q.split,
                success=reference_match and predicates_passed,
                reference_match=reference_match,
                predicates_passed=predicates_passed,
                l_inf=l_inf,
                lpips=lpips,
                solver_note=plan.note,
                expected_skill_ids=q.expected_skill_ids,
                used_skill_ids=execution.used_skill_ids,
                predicate_results=predicate_results,
            )
        )

    return DownstreamResult(
        scores=tuple(scores),
        meta={
            "phase": 7,
            "kind": "downstream_eval",
            "n_queries": len(scores),
            "solver": solver.name,
            "seed": seed,
            "perceptual_backend": getattr(perceptual, "name", None),
            "thresholds_calibrated": thresholds.calibrated,
            **(meta or {}),
        },
    )

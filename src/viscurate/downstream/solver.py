"""Downstream solver agents and trusted execution of their plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from viscurate.downstream.query import Query, QueryStep
from viscurate.skills.model import Image
from viscurate.skills.registry import SkillRegistry

__all__ = [
    "ExpectedSkillSolver",
    "KeywordRetrievalSolver",
    "NoOpSolver",
    "PlanExecution",
    "SolverAgent",
    "SolverPlan",
    "execute_plan",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SolverPlan(_Frozen):
    """The solver's proposed skill pipeline for one query."""

    query_id: str
    steps: tuple[QueryStep, ...] = ()
    note: str = ""

    @property
    def used_skill_ids(self) -> tuple[str, ...]:
        return tuple(step.skill_id for step in self.steps)


@runtime_checkable
class SolverAgent(Protocol):
    """Maps a natural-language query + library to a skill plan."""

    name: str

    def solve(self, query: Query, registry: SkillRegistry) -> SolverPlan: ...


class ExpectedSkillSolver:
    """Upper-bound solver that replays the manifest's clean reference pipeline."""

    name = "expected-skill-oracle"

    def solve(self, query: Query, registry: SkillRegistry) -> SolverPlan:
        del registry
        return SolverPlan(query_id=query.query_id, steps=query.pipeline, note="manifest pipeline")


class NoOpSolver:
    """Baseline solver that returns the input unchanged."""

    name = "no-op"

    def solve(self, query: Query, registry: SkillRegistry) -> SolverPlan:
        del registry
        return SolverPlan(query_id=query.query_id, steps=(), note="no skill selected")


class KeywordRetrievalSolver:
    """Simple dependency-free retrieval over skill id/name/description text."""

    name = "keyword-retrieval"

    def solve(self, query: Query, registry: SkillRegistry) -> SolverPlan:
        q_tokens = _tokens(query.instruction)
        best_id = ""
        best_score = 0
        for skill in registry.all():
            text = f"{skill.id} {skill.name} {skill.description} {skill.metadata.family}"
            score = len(q_tokens & _tokens(text))
            if score > best_score:
                best_score = score
                best_id = skill.id
        if not best_id:
            return SolverPlan(query_id=query.query_id, note="no lexical match")
        return SolverPlan(
            query_id=query.query_id,
            steps=(
                QueryStep(
                    skill_id=best_id,
                    params=registry.get(best_id).params_schema.defaults(),
                ),
            ),
            note=f"token-overlap={best_score}",
        )


@dataclass(frozen=True)
class PlanExecution:
    """Result of executing a solver plan."""

    ok: bool
    output: Image | None
    error: str = ""
    used_skill_ids: tuple[str, ...] = ()


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) >= 2}


def execute_plan(
    registry: SkillRegistry,
    plan: SolverPlan,
    input_image: Image,
    *,
    seed: int = 0,
    allow_untrusted: bool = False,
) -> PlanExecution:
    """Execute a solver plan; untrusted skills require an explicit sandbox-backed opt-in."""
    out = input_image
    used: list[str] = []
    for step in plan.steps:
        if step.skill_id not in registry:
            return PlanExecution(
                ok=False,
                output=None,
                error=f"unknown skill id: {step.skill_id}",
                used_skill_ids=tuple(used),
            )
        skill = registry.get(step.skill_id)
        if not skill.metadata.trusted and not allow_untrusted:
            return PlanExecution(
                ok=False,
                output=None,
                error=f"blocked untrusted skill: {step.skill_id}",
                used_skill_ids=tuple(used),
            )
        try:
            out = skill.run(out, dict(step.params), seed=seed)
        except Exception as exc:
            return PlanExecution(
                ok=False,
                output=None,
                error=f"{type(exc).__name__}: {exc}"[:200],
                used_skill_ids=tuple(used),
            )
        used.append(step.skill_id)
    return PlanExecution(ok=True, output=out, used_skill_ids=tuple(used))

"""The curation environment — the agent loop's engine (CLAUDE.md §3.2, §3.5.7).

``CurationEnvironment`` holds the mutable library being curated, the output-grounded verifier,
usage statistics, and the sandbox trust boundary. It exposes:

* :meth:`observe` → a :class:`CurationState` (library summaries with **no internal labels**,
  usage, recent history, remaining budget);
* :meth:`apply` → applies one :class:`Action` and returns a structured :class:`ActionResult`,
  enforcing the **hard verifier gate** on structural edits and the **trust gate** on
  agent-generated code, logging every action.

The verifier/agent split is honoured exactly (CLAUDE.md §3.2): the environment is the only
place the two meet. It hands the verifier a ``ComparatorView`` + an ``OutputProvider`` (never a
description), gates ``merge`` / ``parameterize`` on the certifying relation, and returns the
structured rejection of §3.5.7 when the relation does not license the edit. ``add`` / ``modify``
to an ``fn`` would produce untrusted code — those stay blocked pending the hardened sandbox
(:mod:`viscurate.curation.sandbox`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from viscurate.config import ThresholdConfig
from viscurate.curation.actions import Action, ActionKind, ActionResult, ActionStatus
from viscurate.curation.gating import gate_structural
from viscurate.curation.sandbox import REVIEW_REQUIRED, ExecutionPolicy
from viscurate.curation.state import CurationState, UsageStats
from viscurate.equivalence.backends import PerceptualBackend, SemanticBackend
from viscurate.equivalence.compare import BatteryEvaluator, OutputProvider
from viscurate.equivalence.param_alignment import ParamAlignment
from viscurate.equivalence.taxonomy import classify
from viscurate.logging import get_logger
from viscurate.skills.model import Image, ParamSpec, ParamsSchema, Skill, SkillMetadata
from viscurate.skills.registry import SkillRegistry

__all__ = ["CurationEnvironment", "EpisodeResult", "run_episode"]


def _untrusted_stub(image: Image, params: dict[str, Any], seed: int) -> Image:  # pragma: no cover
    """Placeholder fn for an agent-added skill — never executed (the skill is ``trusted=False``)."""
    raise RuntimeError("untrusted agent-generated skill cannot execute (hardened sandbox pending)")


def _replace_default(schema: ParamsSchema, param_name: str, value: object) -> ParamsSchema:
    """Return ``schema`` with one parameter's default replaced, re-validated."""
    out: list[ParamSpec] = []
    found = False
    for p in schema.params:
        if p.name == param_name:
            found = True
            out.append(
                ParamSpec(
                    name=p.name,
                    type=p.type,
                    default=value,
                    minimum=p.minimum,
                    maximum=p.maximum,
                    choices=p.choices,
                    description=p.description,
                )
            )
        else:
            out.append(p)
    if not found:
        raise KeyError(param_name)
    return ParamsSchema(params=tuple(out))


class CurationEnvironment:
    """A curatable library + the verifier gate + the trust boundary + an action log."""

    def __init__(
        self,
        registry: SkillRegistry,
        provider: OutputProvider,
        *,
        thresholds: ThresholdConfig,
        usage: UsageStats | None = None,
        alignment: ParamAlignment | None = None,
        perceptual: PerceptualBackend | None = None,
        semantic: SemanticBackend | None = None,
        clip: SemanticBackend | None = None,
        policy: ExecutionPolicy | None = None,
        budget: int = 50,
        usage_fold_threshold: int = 1,
        logger: Any | None = None,
    ) -> None:
        self._registry = registry
        self._provider = provider
        self._thresholds = thresholds
        self._usage = usage or UsageStats.empty()
        self._alignment = alignment
        self._perceptual = perceptual
        self._semantic = semantic
        self._clip = clip
        self._policy = policy or ExecutionPolicy()
        self._budget = budget
        self._usage_fold_threshold = usage_fold_threshold
        self._log = logger or get_logger("curation")
        self._history: list[ActionResult] = []
        self._actions_taken = 0
        # Skills the verifier can actually execute (trusted, present in the provider).
        self._verifiable: set[str] = {s.id for s in registry.all() if s.metadata.trusted}

    # -- accessors ---------------------------------------------------------------------
    @property
    def registry(self) -> SkillRegistry:
        return self._registry

    @property
    def history(self) -> tuple[ActionResult, ...]:
        return tuple(self._history)

    @property
    def actions_taken(self) -> int:
        return self._actions_taken

    @property
    def budget_remaining(self) -> int:
        return max(0, self._budget - self._actions_taken)

    def observe(self) -> CurationState:
        return CurationState.build(
            self._registry.all(),
            self._usage,
            actions_taken=self._actions_taken,
            budget_remaining=self.budget_remaining,
            history=self._history,
        )

    # -- the dispatcher ----------------------------------------------------------------
    def apply(self, action: Action) -> ActionResult:
        """Apply one action, enforcing the verifier + trust gates, and log the outcome."""
        before = len(self._registry)
        result = self._dispatch(action, before)
        if action.kind is not ActionKind.END:
            self._actions_taken += 1
        self._history.append(result)
        self._log.info(
            "curation_action",
            kind=action.kind.value,
            primary=action.primary,
            secondary=action.secondary,
            status=result.status.value,
            relation=result.relation,
            size_before=result.size_before,
            size_after=result.size_after,
        )
        return result

    def _dispatch(self, action: Action, before: int) -> ActionResult:
        kind = action.kind
        if kind is ActionKind.END:
            return self._result(action, ActionStatus.NOOP, before, before, reason="episode ended")
        if kind is ActionKind.RETRIEVE:
            target = action.query or action.primary
            return self._result(
                action, ActionStatus.NOOP, before, before, reason=f"retrieve({target}) recorded"
            )
        if kind is ActionKind.MERGE or kind is ActionKind.PARAMETERIZE:
            return self._structural(action, before)
        if kind is ActionKind.SPLIT:
            # Splitting requires verifying agent-authored specialization fns, which are untrusted
            # in v1 — blocked pending the hardened sandbox (CLAUDE.md §5).
            return self._result(
                action,
                ActionStatus.BLOCKED,
                before,
                before,
                reason="split requires verifying agent-authored specializations; "
                + REVIEW_REQUIRED,
            )
        if kind is ActionKind.REMOVE:
            return self._remove(action, before)
        if kind is ActionKind.MODIFY:
            return self._modify(action, before)
        if kind is ActionKind.ADD:
            return self._add(action, before)
        raise ValueError(f"unhandled action kind {kind!r}")  # pragma: no cover

    # -- structural edits (verifier-gated) ---------------------------------------------
    def _structural(self, action: Action, before: int) -> ActionResult:
        a, b = action.primary, action.secondary
        if not a or not b or a == b:
            return self._result(
                action,
                ActionStatus.INVALID,
                before,
                before,
                reason="merge/parameterize need two distinct skill ids (primary, secondary)",
            )
        if a not in self._registry or b not in self._registry:
            missing = [x for x in (a, b) if x not in self._registry]
            return self._result(
                action,
                ActionStatus.INVALID,
                before,
                before,
                reason=f"unknown skill id(s): {missing}",
            )
        # Trust gate: the verifier executes both skills in-process; untrusted code is blocked.
        for sid in (a, b):
            decision = self._policy.gate(trusted=self._is_verifiable(sid))
            if not decision.permitted:
                return self._result(
                    action,
                    ActionStatus.BLOCKED,
                    before,
                    before,
                    reason=f"cannot verify {sid}: {decision.reason}",
                )

        result = classify(
            self._provider.comparator_view(a),
            self._provider.comparator_view(b),
            self._provider,
            thresholds=self._thresholds,
            perceptual=self._perceptual,
            semantic=self._semantic,
            clip=self._clip,
            alignment=self._alignment,
        )
        gate = gate_structural(action, result)
        if not gate.permitted:
            return self._result(
                action,
                ActionStatus.REJECTED,
                before,
                before,
                reason=gate.reason,
                relation=gate.relation.value,
                direction=gate.direction.value,
                distances=gate.distances,
                alternatives=gate.alternatives,
            )

        # Permitted: remove the primary (duplicate/specialization); keep the survivor (secondary).
        self._registry.remove(a)
        self._verifiable.discard(a)
        reason = gate.reason
        if action.kind is ActionKind.PARAMETERIZE and self._usage.is_heavily_used(
            a, threshold=self._usage_fold_threshold
        ):
            reason += f" (note: {a} has usage {self._usage.usage(a)} — folding loses a used skill)"
        return self._result(
            action,
            ActionStatus.APPLIED,
            before,
            len(self._registry),
            reason=reason,
            relation=gate.relation.value,
            direction=gate.direction.value,
            distances=gate.distances,
            alternatives=gate.alternatives,
        )

    # -- non-structural edits ----------------------------------------------------------
    def _remove(self, action: Action, before: int) -> ActionResult:
        sid = action.primary
        if sid not in self._registry:
            return self._result(
                action, ActionStatus.INVALID, before, before, reason=f"unknown skill id: {sid!r}"
            )
        used = self._usage.usage(sid)
        referenced = self._usage.is_referenced(sid)
        self._registry.remove(sid)
        self._verifiable.discard(sid)
        note = f"removed {sid} (usage={used}, referenced={referenced})"
        if referenced:
            note += " — WARNING: removing a referenced/used skill loses functional coverage"
        return self._result(action, ActionStatus.APPLIED, before, len(self._registry), reason=note)

    def _modify(self, action: Action, before: int) -> ActionResult:
        sid = action.primary
        if sid not in self._registry:
            return self._result(
                action, ActionStatus.INVALID, before, before, reason=f"unknown skill id: {sid!r}"
            )
        skill = self._registry.get(sid)
        name = action.new_name or skill.name
        description = action.new_description or skill.description
        schema = skill.params_schema
        if action.param_name:
            try:
                schema = _replace_default(schema, action.param_name, action.value)
            except (KeyError, ValueError) as exc:
                return self._result(
                    action,
                    ActionStatus.INVALID,
                    before,
                    before,
                    reason=f"invalid param edit for {sid}: {exc}",
                )
        if (
            name == skill.name
            and description == skill.description
            and schema is skill.params_schema
        ):
            return self._result(
                action,
                ActionStatus.NOOP,
                before,
                before,
                reason="modify is a no-op (no metadata/schema change; fn edits need the sandbox)",
            )
        # Output-preserving repair (name/description/schema default) keeps the fn → stays trusted.
        self._registry.register(
            Skill(
                id=skill.id,
                name=name,
                description=description,
                fn=skill.fn,
                params_schema=schema,
                metadata=skill.metadata,
            ),
            replace=True,
        )
        return self._result(
            action,
            ActionStatus.APPLIED,
            before,
            len(self._registry),
            reason=f"modified {sid} (metadata/schema; fn unchanged)",
        )

    def _add(self, action: Action, before: int) -> ActionResult:
        sid = action.new_skill_id
        if not sid:
            return self._result(
                action, ActionStatus.INVALID, before, before, reason="add needs new_skill_id"
            )
        if sid in self._registry:
            return self._result(
                action,
                ActionStatus.INVALID,
                before,
                before,
                reason=f"skill id already exists: {sid!r}",
            )
        # Agent-generated → trusted=False, blocked from execution until the hardened sandbox is
        # reviewed (CLAUDE.md §5). Registered so it appears in the library, never verifiable/run.
        self._registry.register(
            Skill(
                id=sid,
                name=action.new_name or sid,
                description=action.new_description,
                fn=_untrusted_stub,
                params_schema=ParamsSchema(),
                metadata=SkillMetadata(
                    family=action.family or "agent", provenance="agent", trusted=False
                ),
            )
        )
        return self._result(
            action,
            ActionStatus.APPLIED,
            before,
            len(self._registry),
            reason=f"added {sid} as trusted=False (blocked from execution; {REVIEW_REQUIRED})",
        )

    # -- helpers -----------------------------------------------------------------------
    def _is_verifiable(self, skill_id: str) -> bool:
        return skill_id in self._verifiable

    def _result(
        self,
        action: Action,
        status: ActionStatus,
        size_before: int,
        size_after: int,
        *,
        reason: str = "",
        relation: str = "",
        direction: str = "",
        distances: dict[str, float] | None = None,
        alternatives: tuple[str, ...] = (),
    ) -> ActionResult:
        return ActionResult(
            action=action,
            status=status,
            reason=reason,
            relation=relation,
            direction=direction,
            distances=distances or {},
            alternatives=alternatives,
            size_before=size_before,
            size_after=size_after,
        )

    @classmethod
    def from_skills(
        cls,
        skills: Sequence[Skill],
        battery: Sequence[tuple[str, Image]],
        *,
        thresholds: ThresholdConfig,
        seed: int = 0,
        **kwargs: Any,
    ) -> CurationEnvironment:
        """Build the environment + its verifier evaluator over ``skills`` (the L_ρ library)."""
        registry = SkillRegistry()
        for s in skills:
            registry.register(s)
        provider = BatteryEvaluator(list(skills), battery, seed=seed)
        return cls(registry, provider, thresholds=thresholds, **kwargs)


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one curation episode: the action log + size/compression summary."""

    log: tuple[ActionResult, ...]
    size_before: int
    size_after: int
    ended: bool

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in ActionStatus}
        for r in self.log:
            out[r.status.value] += 1
        return out

    def applied_kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.log:
            if r.applied:
                out[r.action.kind.value] = out.get(r.action.kind.value, 0) + 1
        return out

    @property
    def compression(self) -> int:
        """Skills removed over the episode (the library-compression Pareto axis, CLAUDE.md §3.4)."""
        return self.size_before - self.size_after


def run_episode(
    env: CurationEnvironment, agent: Any, *, max_steps: int | None = None
) -> EpisodeResult:
    """Drive ``agent`` against ``env`` until it ends, the budget is spent, or ``max_steps``.

    ``agent`` must implement ``propose(state) -> Action`` (the :class:`CurationAgent` protocol).
    """
    size_before = len(env.registry)
    ended = False
    steps = 0
    while env.budget_remaining > 0 and (max_steps is None or steps < max_steps):
        action = agent.propose(env.observe())
        env.apply(action)
        steps += 1
        if action.kind is ActionKind.END:
            ended = True
            break
    return EpisodeResult(
        log=env.history,
        size_before=size_before,
        size_after=len(env.registry),
        ended=ended,
    )

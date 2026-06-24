"""The observable curation state handed to the agent (CLAUDE.md §3.2, §3.5.7).

The agent observes *library summaries with no internal labels*, usage statistics, and the
action history. The no-internal-label boundary is structural: a :class:`SkillSummary` is built
from :meth:`SkillMetadata.agent_view`, which omits ``is_buggy`` / ``is_dead`` — so a buggy or
dead skill is indistinguishable from a clean one in the state, exactly as the dataset intends
(the agent must *discover* defects from outputs and usage, CLAUDE.md §1.2).

Usage is the Layer-E seam (CLAUDE.md §2.4): Phase 6 accepts a :class:`UsageStats` from any
source (empty, hand-specified, or a synthetic Zipfian log); the query-driven usage that makes
"should we merge?" diverge from "can we merge?" lands in Phase 7.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from viscurate.curation.actions import ActionResult
from viscurate.skills.model import Skill

__all__ = [
    "CurationState",
    "ParamSummary",
    "SkillSummary",
    "UsageStats",
]


@dataclass(frozen=True)
class UsageStats:
    """Per-skill usage counts + query relevance (the Layer-E seam, CLAUDE.md §2.4).

    ``counts`` is a synthetic/observed usage frequency per skill id; ``referenced`` is the set
    of skill ids some query is known to need (non-zero retrieval relevance). A skill is *dead*
    from the agent's vantage point when it is neither used nor referenced — the only signal the
    verifier cannot see (CLAUDE.md §3.5.8).
    """

    counts: Mapping[str, int] = field(default_factory=dict)
    referenced: frozenset[str] = field(default_factory=frozenset)

    def usage(self, skill_id: str) -> int:
        return int(self.counts.get(skill_id, 0))

    def is_referenced(self, skill_id: str) -> bool:
        return skill_id in self.referenced or self.usage(skill_id) > 0

    def is_heavily_used(self, skill_id: str, *, threshold: int = 1) -> bool:
        """Whether a skill is used enough that folding it away would hurt (the §3.5.7 gate)."""
        return self.usage(skill_id) >= threshold

    @classmethod
    def empty(cls) -> UsageStats:
        return cls()


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParamSummary(_Frozen):
    """A skill parameter as the agent sees it: name, type, default (no internal detail)."""

    name: str
    type: str
    default: object | None = None


class SkillSummary(_Frozen):
    """What the agent sees about one skill — **never** the internal ground-truth labels.

    Built via :meth:`from_skill`, which reads :meth:`SkillMetadata.agent_view` so ``is_buggy`` /
    ``is_dead`` are absent by construction (CLAUDE.md §1.2).
    """

    id: str
    name: str
    description: str
    family: str
    provenance: str
    trusted: bool
    seeded_stochastic: bool
    precision_sensitive: bool
    platform_sensitive: bool
    params: tuple[ParamSummary, ...] = ()
    usage_count: int = 0
    referenced: bool = False

    @classmethod
    def from_skill(cls, skill: Skill, usage: UsageStats) -> SkillSummary:
        view = skill.metadata.agent_view()  # excludes is_buggy / is_dead
        return cls(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            family=str(view["family"]),
            provenance=str(view["provenance"]),
            trusted=bool(view["trusted"]),
            seeded_stochastic=bool(view["seeded_stochastic"]),
            precision_sensitive=bool(view["precision_sensitive"]),
            platform_sensitive=bool(view["platform_sensitive"]),
            params=tuple(
                ParamSummary(name=p.name, type=p.type, default=p.default)
                for p in skill.params_schema.params
            ),
            usage_count=usage.usage(skill.id),
            referenced=usage.is_referenced(skill.id),
        )


class CurationState(_Frozen):
    """A snapshot the agent reasons over: skills, usage, recent history, remaining budget."""

    skills: tuple[SkillSummary, ...]
    actions_taken: int
    budget_remaining: int
    history: tuple[ActionResult, ...] = ()

    @property
    def size(self) -> int:
        return len(self.skills)

    @classmethod
    def build(
        cls,
        skills: Sequence[Skill],
        usage: UsageStats,
        *,
        actions_taken: int,
        budget_remaining: int,
        history: Sequence[ActionResult] = (),
    ) -> CurationState:
        return cls(
            skills=tuple(SkillSummary.from_skill(s, usage) for s in skills),
            actions_taken=actions_taken,
            budget_remaining=budget_remaining,
            history=tuple(history),
        )

    def render(self, *, max_history: int = 8) -> str:
        """A compact text view for an LLM agent (the observation prompt)."""
        lines = [
            f"Library: {self.size} skills. "
            f"Budget remaining: {self.budget_remaining} actions "
            f"({self.actions_taken} taken).",
            "",
            "Skills (id | family | usage | description):",
        ]
        for s in self.skills:
            flags = []
            if not s.trusted:
                flags.append("UNTRUSTED")
            if s.provenance != "builtin":
                flags.append(s.provenance)
            tag = f" [{','.join(flags)}]" if flags else ""
            lines.append(
                f"  {s.id} | {s.family} | used={s.usage_count} | {s.description[:80]}{tag}"
            )
        if self.history:
            lines += ["", "Recent actions:"]
            for r in self.history[-max_history:]:
                detail = f"{r.action.primary}" + (
                    f"->{r.action.secondary}" if r.action.secondary else ""
                )
                lines.append(f"  {r.action.kind}({detail}) -> {r.status}: {r.reason[:80]}")
        return "\n".join(lines)

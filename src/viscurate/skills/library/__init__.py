"""The built-in skill library — the clean base ``L0`` (CLAUDE.md §1.1).

These are defect-free: no bugs, no duplicates, no dead skills. Corruption (the seven
defect types) is *injected later* at a controlled rate ρ (Phase 5); ``L0`` stays clean.

This batch covers the geometric, colour, and signal/blur/edge/morphology families — enough
to exercise the harness and to host the planted relations (subsumption pairs, hard
negatives, a complementary pair, a seeded-stochastic skill) the benchmark depends on. The
remaining families (masks / reconstruction / synthesis) land later in Phase 1.
"""

from __future__ import annotations

from viscurate.skills.library import color, filtering, geometric, reconstruction
from viscurate.skills.model import Skill, SkillFn
from viscurate.skills.registry import SkillRegistry

__all__ = [
    "build_builtin_registry",
    "builtin_fn_resolver",
    "builtin_skills",
    "load_builtin_skills",
]


def builtin_skills() -> list[Skill]:
    """Return freshly-built instances of every built-in skill (clean ``L0``)."""
    skills: list[Skill] = []
    for module in (geometric, color, filtering, reconstruction):
        skills.extend(module.build())
    _check_unique_ids(skills)
    return skills


def _check_unique_ids(skills: list[Skill]) -> None:
    seen: set[str] = set()
    for s in skills:
        if s.id in seen:
            raise ValueError(f"duplicate built-in skill id: {s.id!r}")
        seen.add(s.id)


def load_builtin_skills(registry: SkillRegistry) -> SkillRegistry:
    """Register every built-in skill into ``registry`` and return it."""
    for skill in builtin_skills():
        registry.register(skill)
    return registry


def build_builtin_registry() -> SkillRegistry:
    """Return a fresh registry populated with the built-in library."""
    return load_builtin_skills(SkillRegistry())


def builtin_fn_resolver(skill_id: str) -> SkillFn:
    """Resolve a skill id to its built-in callable (for registry deserialization)."""
    table = _fn_table()
    if skill_id not in table:
        raise KeyError(f"no built-in fn for skill id {skill_id!r}")
    return table[skill_id]


_FN_TABLE: dict[str, SkillFn] | None = None


def _fn_table() -> dict[str, SkillFn]:
    global _FN_TABLE
    if _FN_TABLE is None:
        _FN_TABLE = {s.id: s.fn for s in builtin_skills()}
    return _FN_TABLE

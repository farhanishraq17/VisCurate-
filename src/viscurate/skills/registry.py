"""A JSON-serializable registry of skills (CLAUDE.md Phase 1 deliverable).

Persistence stores :class:`SkillSpec` records (no callables). Reloading re-binds each
skill's ``fn`` via a *resolver* keyed by skill id — for the built-in library the resolver
is the function table in :mod:`viscurate.skills.library`. Skill ids are stable and never
reused, so the resolver mapping is a durable contract.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

from viscurate.skills.model import Skill, SkillFn, SkillSpec

__all__ = ["FnResolver", "SkillRegistry"]

FnResolver = Callable[[str], SkillFn]

_SCHEMA_VERSION = "1"


class SkillRegistry:
    """An ordered, id-unique collection of :class:`Skill`."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    # -- mutation -----------------------------------------------------------------
    def register(self, skill: Skill, *, replace: bool = False) -> Skill:
        if skill.id in self._skills and not replace:
            raise ValueError(f"skill id already registered: {skill.id!r}")
        self._skills[skill.id] = skill
        return skill

    def remove(self, skill_id: str) -> None:
        del self._skills[skill_id]

    # -- access -------------------------------------------------------------------
    def get(self, skill_id: str) -> Skill:
        return self._skills[skill_id]

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def ids(self) -> list[str]:
        return list(self._skills)

    def __contains__(self, skill_id: object) -> bool:
        return skill_id in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())

    # -- serialization ------------------------------------------------------------
    def to_specs(self) -> list[SkillSpec]:
        return [s.to_spec() for s in self._skills.values()]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "skills": [spec.model_dump(mode="json") for spec in self.to_specs()],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, object], resolver: FnResolver) -> SkillRegistry:
        version = data.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported registry schema_version {version!r}")
        raw_skills = data.get("skills")
        if not isinstance(raw_skills, list):
            raise TypeError("registry 'skills' must be a list")
        reg = cls()
        for raw in raw_skills:
            spec = SkillSpec.model_validate(raw)
            reg.register(Skill.from_spec(spec, resolver(spec.id)))
        return reg

    @classmethod
    def from_json(cls, text: str, resolver: FnResolver) -> SkillRegistry:
        return cls.from_dict(json.loads(text), resolver)

    @classmethod
    def load(cls, path: str | Path, resolver: FnResolver) -> SkillRegistry:
        return cls.from_json(Path(path).read_text(encoding="utf-8"), resolver)

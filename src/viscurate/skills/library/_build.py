"""Tiny builders to keep skill definitions terse and uniform."""

from __future__ import annotations

from typing import Any

from viscurate.skills.model import (
    ParamSpec,
    ParamsSchema,
    Skill,
    SkillFn,
    SkillMetadata,
)


def param(
    name: str,
    type: str,
    default: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    choices: tuple[Any, ...] | None = None,
    description: str = "",
) -> ParamSpec:
    return ParamSpec(
        name=name,
        type=type,  # type: ignore[arg-type]  # validated by ParamSpec
        default=default,
        minimum=minimum,
        maximum=maximum,
        choices=choices,
        description=description,
    )


def make_skill(
    skill_id: str,
    name: str,
    description: str,
    fn: SkillFn,
    family: str,
    params: tuple[ParamSpec, ...] = (),
    *,
    seeded_stochastic: bool = False,
    precision_sensitive: bool = False,
    platform_sensitive: bool = False,
) -> Skill:
    return Skill(
        id=skill_id,
        name=name,
        description=description,
        fn=fn,
        params_schema=ParamsSchema(params=params),
        metadata=SkillMetadata(
            family=family,
            provenance="builtin",
            trusted=True,
            seeded_stochastic=seeded_stochastic,
            precision_sensitive=precision_sensitive,
            platform_sensitive=platform_sensitive,
        ),
    )

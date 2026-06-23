"""Intentionally-misbehaving skills used to test the executor's failure paths.

These live in an importable package module (not a test file) so the subprocess worker can
always resolve them when unpickling. They are never registered in the built-in library.
"""

from __future__ import annotations

import time

from viscurate.skills.model import (
    Image,
    Params,
    ParamSpec,
    ParamsSchema,
    Skill,
    SkillMetadata,
)


def _sleep_fn(image: Image, params: Params, seed: int) -> Image:
    time.sleep(float(params["seconds"]))
    return image


def _crash_fn(image: Image, params: Params, seed: int) -> Image:
    raise RuntimeError("intentional crash for executor test")


def make_sleep_skill(*, trusted: bool = True) -> Skill:
    return Skill(
        id="_testkit_sleep",
        name="sleep",
        description="Sleeps for `seconds`, to exercise the wall-clock timeout.",
        fn=_sleep_fn,
        params_schema=ParamsSchema(
            params=(ParamSpec(name="seconds", type="float", default=0.0, minimum=0.0),)
        ),
        metadata=SkillMetadata(family="_test", provenance="testkit", trusted=trusted),
    )


def make_crash_skill(*, trusted: bool = True) -> Skill:
    return Skill(
        id="_testkit_crash",
        name="crash",
        description="Raises immediately, to exercise the error path.",
        fn=_crash_fn,
        params_schema=ParamsSchema(),
        metadata=SkillMetadata(family="_test", provenance="testkit", trusted=trusted),
    )

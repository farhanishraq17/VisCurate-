from __future__ import annotations

import numpy as np
import pytest

from viscurate.skills.model import ParamSpec, ParamsSchema, Skill, SkillMetadata


def test_paramspec_coerce_types_and_ranges() -> None:
    p = ParamSpec(name="k", type="int", default=5, minimum=1, maximum=31)
    assert p.coerce(7) == 7
    assert p.coerce(7.0) == 7  # integral float accepted, coerced to int
    with pytest.raises(ValueError):
        p.coerce(7.5)
    with pytest.raises(ValueError):
        p.coerce(0)  # below minimum
    with pytest.raises(ValueError):
        p.coerce(True)  # bool is not an int here


def test_enum_requires_choices_and_validates() -> None:
    with pytest.raises(ValueError):
        ParamSpec(name="m", type="enum", default="a")  # no choices
    p = ParamSpec(name="m", type="enum", default="a", choices=("a", "b"))
    assert p.coerce("b") == "b"
    with pytest.raises(ValueError):
        p.coerce("c")


def test_invalid_default_is_rejected() -> None:
    with pytest.raises(ValueError):
        ParamSpec(name="k", type="int", default=99, minimum=0, maximum=10)


def test_params_schema_fills_defaults_and_rejects_unknown() -> None:
    schema = ParamsSchema(
        params=(
            ParamSpec(name="a", type="int", default=1),
            ParamSpec(name="b", type="float", default=2.0),
        )
    )
    assert schema.validate_params(None) == {"a": 1, "b": 2.0}
    assert schema.validate_params({"a": 5}) == {"a": 5, "b": 2.0}
    with pytest.raises(ValueError):
        schema.validate_params({"c": 1})


def test_agent_view_hides_internal_labels() -> None:
    meta = SkillMetadata(family="blur", is_buggy=True, is_dead=True)
    view = meta.agent_view()
    assert "is_buggy" not in view
    assert "is_dead" not in view
    assert view["family"] == "blur"


def test_comparator_view_has_no_description() -> None:
    skill = Skill(
        id="x",
        name="X",
        description="SECRET TEXT the comparator must never read",
        fn=lambda img, params, seed: img,
        params_schema=ParamsSchema(),
        metadata=SkillMetadata(family="t"),
    )
    cview = skill.comparator_view()
    assert not hasattr(cview, "description")
    assert cview.id == "x"


def test_skill_run_validates_and_calls() -> None:
    skill = Skill(
        id="add1",
        name="add1",
        description="",
        fn=lambda img, params, seed: img + 1,
        params_schema=ParamsSchema(),
        metadata=SkillMetadata(family="t"),
    )
    out = skill.run(np.zeros((2, 2), dtype=np.uint8))
    assert np.array_equal(out, np.ones((2, 2), dtype=np.uint8))
    with pytest.raises(TypeError):
        skill.run([1, 2, 3])  # type: ignore[arg-type]

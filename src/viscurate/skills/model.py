"""The ``Skill`` model and its JSON-serializable spec (CLAUDE.md §1.2).

A skill bundles a deterministic ``fn(image, params, seed) -> image`` with typed parameter
metadata. Two boundaries are encoded structurally rather than by convention:

* ``description`` is for **text/embedding baselines only**; the output-grounded comparator
  path must never read it. :meth:`Skill.comparator_view` is the object the verifier is
  handed — it has no ``description`` attribute at all, so a comparator that reaches for one
  fails loudly.
* ``is_buggy`` / ``is_dead`` are **internal-only** ground-truth labels and are never shown
  to the agent. :meth:`SkillMetadata.agent_view` omits them.

The callable ``fn`` is not JSON-serializable, so persistence round-trips a
:class:`SkillSpec` (everything *but* ``fn``) and re-binds ``fn`` from a resolver keyed by
skill id (see :mod:`viscurate.skills.registry`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Image",
    "ParamSpec",
    "Params",
    "ParamsSchema",
    "Skill",
    "SkillFn",
    "SkillMetadata",
    "SkillSpec",
]

# An image is an HxW, HxWx3 or HxWx4 array. We keep the dtype open (uint8 by contract on
# input, but skills may return masks/edge maps) and canonicalize downstream (§1.3).
Image = npt.NDArray[Any]
Params = dict[str, Any]
SkillFn = Callable[[Image, Params, int], Image]

ParamType = Literal["int", "float", "bool", "str", "enum"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParamSpec(_Frozen):
    """A single typed parameter: default + valid range/choices (CLAUDE.md §1.2)."""

    name: str
    type: ParamType
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> ParamSpec:
        if self.type == "enum" and not self.choices:
            raise ValueError(f"enum param {self.name!r} must declare choices")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"param {self.name!r}: minimum > maximum")
        # The default must itself be valid.
        self.coerce(self.default)
        return self

    def coerce(self, value: Any) -> Any:
        """Validate and coerce ``value`` to this parameter's type/range; raise on failure."""
        v = value
        if self.type == "int":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
                raise ValueError(f"param {self.name!r} expects int, got {value!r}")
            v = int(v)
        elif self.type == "float":
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"param {self.name!r} expects float, got {value!r}")
            v = float(v)
        elif self.type == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"param {self.name!r} expects bool, got {value!r}")
        elif self.type == "str":
            if not isinstance(v, str):
                raise ValueError(f"param {self.name!r} expects str, got {value!r}")
        elif self.type == "enum":
            if self.choices is None or v not in self.choices:
                raise ValueError(
                    f"param {self.name!r} must be one of {self.choices}, got {value!r}"
                )
            return v

        if self.minimum is not None and float(v) < self.minimum:
            raise ValueError(f"param {self.name!r} = {v} below minimum {self.minimum}")
        if self.maximum is not None and float(v) > self.maximum:
            raise ValueError(f"param {self.name!r} = {v} above maximum {self.maximum}")
        return v


class ParamsSchema(_Frozen):
    """An ordered collection of :class:`ParamSpec`, with default-fill + validation."""

    params: tuple[ParamSpec, ...] = ()

    @model_validator(mode="after")
    def _unique_names(self) -> ParamsSchema:
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate parameter names: {names}")
        return self

    def defaults(self) -> Params:
        return {p.name: p.default for p in self.params}

    def validate_params(self, given: Params | None) -> Params:
        """Fill defaults for missing keys, coerce/range-check provided keys, reject unknowns."""
        given = dict(given or {})
        known = {p.name: p for p in self.params}
        unknown = set(given) - set(known)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        out: Params = {}
        for name, spec in known.items():
            out[name] = spec.coerce(given[name]) if name in given else spec.default
        return out


class SkillMetadata(_Frozen):
    """Family tag, provenance, determinism flags, and internal-only ground-truth labels."""

    family: str
    provenance: str = "builtin"
    # Lightweight isolation gate: only trusted skills may execute (CLAUDE.md D6).
    trusted: bool = True
    # Determinism classification (CLAUDE.md §1.4).
    seeded_stochastic: bool = False
    precision_sensitive: bool = False
    platform_sensitive: bool = False
    # INTERNAL-ONLY ground-truth labels — never shown to the agent (CLAUDE.md §1.2).
    is_buggy: bool = False
    is_dead: bool = False

    _INTERNAL_ONLY = ("is_buggy", "is_dead")

    def agent_view(self) -> dict[str, Any]:
        """Metadata safe to expose to the curation agent (no internal labels)."""
        return self.model_dump(exclude=set(self._INTERNAL_ONLY))


class SkillSpec(_Frozen):
    """JSON-serializable description of a skill (everything except the callable ``fn``)."""

    id: str
    name: str
    description: str = ""
    params_schema: ParamsSchema = Field(default_factory=ParamsSchema)
    metadata: SkillMetadata


@dataclass(frozen=True)
class ComparatorView:
    """What the output-grounded verifier is handed: identity + outputs-relevant flags only.

    Deliberately carries **no** ``description`` — the load-bearing modality boundary
    (CLAUDE.md §1.2) is enforced by the type, not by reviewer discipline.
    """

    id: str
    family: str
    precision_sensitive: bool
    platform_sensitive: bool
    seeded_stochastic: bool


@dataclass(frozen=True)
class Skill:
    """A registered skill: typed metadata + a deterministic image→image callable."""

    id: str
    name: str
    description: str
    fn: SkillFn
    params_schema: ParamsSchema
    metadata: SkillMetadata

    def to_spec(self) -> SkillSpec:
        return SkillSpec(
            id=self.id,
            name=self.name,
            description=self.description,
            params_schema=self.params_schema,
            metadata=self.metadata,
        )

    @classmethod
    def from_spec(cls, spec: SkillSpec, fn: SkillFn) -> Skill:
        return cls(
            id=spec.id,
            name=spec.name,
            description=spec.description,
            fn=fn,
            params_schema=spec.params_schema,
            metadata=spec.metadata,
        )

    def comparator_view(self) -> ComparatorView:
        """Hand this (never the :class:`Skill`) to output-grounded comparators."""
        m = self.metadata
        return ComparatorView(
            id=self.id,
            family=m.family,
            precision_sensitive=m.precision_sensitive,
            platform_sensitive=m.platform_sensitive,
            seeded_stochastic=m.seeded_stochastic,
        )

    def run(self, image: Image, params: Params | None = None, seed: int = 0) -> Image:
        """Validate params and call ``fn`` **in-process** (no sandbox).

        This is the trusted bulk path (e.g. oracle freeze). The trusted-gate + timeout +
        resource-limited path is :class:`viscurate.skills.executor.SandboxedExecutor`.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be np.ndarray, got {type(image).__name__}")
        validated = self.params_schema.validate_params(params)
        out = self.fn(image, validated, int(seed))
        if not isinstance(out, np.ndarray):
            raise TypeError(f"skill {self.id!r} returned {type(out).__name__}, expected np.ndarray")
        return out

"""Matched-sweep parameter alignment (CLAUDE.md §3.5.2, §3.5.10.2).

Two parameterized skills rarely share a schema, so "run both and compare" is underspecified.
The fix is an **auditable shared-axis map** — never hard-coded in the comparator. For each
shared *semantic axis* (e.g. kernel size ``k``) the artifact records, per member skill, how an
axis value maps to that skill's params. The comparator then evaluates both skills at matched
axis values and requires the relation to hold **at every grid point** — which is exactly why
``blur_gaussian`` vs ``blur_box`` correctly returns DISTINCT (they agree at small ``k`` and
diverge at large ``k``) where a default-only check would wrongly merge them.

A member maps axis values to params in one of two ways:

* **driven** — one param takes ``value * scale`` and the rest are ``fixed`` (e.g. ``ksize=k``);
* **bindings** — an explicit param dict per axis value (for multi-param specializations such as
  a centered ``crop_bounding_box`` reproducing ``crop_center_percentage``).

The artifact also feeds directional subsumption: an axis grid is the natural search grid for a
generalizing skill (CLAUDE.md §3.5.4).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from viscurate.skills.model import Params

__all__ = ["AlignedAxis", "AxisMember", "ParamAlignment", "load_param_alignment"]


def _vkey(v: float) -> str:
    """Canonical string key for an axis value (so ``5`` and ``5.0`` match ``"5"``)."""
    return str(int(v)) if float(v).is_integer() else str(v)


class AxisMember(BaseModel):
    """How one skill realizes a shared axis: a driven param, or explicit per-value bindings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    param: str | None = None
    scale: float = 1.0
    fixed: dict[str, Any] = {}
    bindings: dict[str, dict[str, Any]] = {}

    @model_validator(mode="after")
    def _one_mode(self) -> AxisMember:
        if (self.param is None) == (not self.bindings):
            raise ValueError("axis member must set exactly one of `param` or `bindings`")
        return self

    def params_for(self, value: float) -> Params | None:
        """The skill's params at axis ``value``, or None if this member skips that value."""
        if self.param is not None:
            return {**self.fixed, self.param: value * self.scale}
        binding = self.bindings.get(_vkey(value))
        if binding is None:
            return None
        return {**self.fixed, **binding}


class AlignedAxis(BaseModel):
    """A shared semantic axis and how each member skill maps its values to params."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    values: tuple[float, ...]
    members: dict[str, AxisMember]

    @model_validator(mode="after")
    def _nonempty(self) -> AlignedAxis:
        if not self.values:
            raise ValueError(f"axis {self.name!r} has no values")
        if len(self.members) < 1:
            raise ValueError(f"axis {self.name!r} has no members")
        return self

    def has(self, skill_id: str) -> bool:
        return skill_id in self.members

    def grid(self, skill_id: str) -> list[Params]:
        """Every params binding this skill takes across the axis (skipping absent values)."""
        member = self.members[skill_id]
        out: list[Params] = []
        for v in self.values:
            p = member.params_for(v)
            if p is not None:
                out.append(p)
        return out

    def matched(self, a_id: str, b_id: str) -> Iterator[tuple[float, Params, Params]]:
        """Yield ``(value, params_a, params_b)`` only where *both* members define a binding."""
        ma, mb = self.members[a_id], self.members[b_id]
        for v in self.values:
            pa, pb = ma.params_for(v), mb.params_for(v)
            if pa is not None and pb is not None:
                yield v, pa, pb


class ParamAlignment(BaseModel):
    """The collection of shared axes (configs/param_alignment.yaml).

    ``axes`` are **symmetric** matched-sweep axes (both members span the axis bidirectionally,
    e.g. ``blur_gaussian``/``blur_box`` over ``ksize``) — used for EXACT/PERCEPTUAL. They are
    deliberately *not* used for asymmetric pairs, where treating a specialization's matched
    binding as full equivalence would mislabel SUBSUMPTION as EXACT.

    ``subsumption_grids`` are explicit per-skill search grids for a *generalizing* skill, so a
    specialization's output is findable during directional subsumption search (e.g.
    ``crop_bounding_box`` carries centered-crop bindings so ``crop_center_percentage`` is
    subsumed). They never enter the EXACT/PERCEPTUAL matched sweep.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "1"
    axes: tuple[AlignedAxis, ...] = ()
    subsumption_grids: dict[str, tuple[dict[str, Any], ...]] = {}

    def subsumption_grid(self, skill_id: str) -> list[Params] | None:
        """Explicit search grid for a generalizing skill, or None if none is configured."""
        grid = self.subsumption_grids.get(skill_id)
        return [dict(p) for p in grid] if grid is not None else None

    def axes_with(self, skill_id: str) -> list[AlignedAxis]:
        return [ax for ax in self.axes if ax.has(skill_id)]

    def axis_for_pair(self, a_id: str, b_id: str) -> AlignedAxis | None:
        """The first axis containing *both* skills (the matched-sweep axis for the pair)."""
        for ax in self.axes:
            if ax.has(a_id) and ax.has(b_id):
                return ax
        return None

    def grid_for(self, skill_id: str) -> list[Params]:
        """Union of this skill's bindings across every axis it participates in (de-duplicated)."""
        seen: set[str] = set()
        out: list[Params] = []
        for ax in self.axes_with(skill_id):
            for p in ax.grid(skill_id):
                key = repr(sorted(p.items()))
                if key not in seen:
                    seen.add(key)
                    out.append(p)
        return out

    def matched_sweep(self, a_id: str, b_id: str) -> list[tuple[Params, Params]] | None:
        """The aligned ``(params_a, params_b)`` grid for a pair, or None if no shared axis."""
        ax = self.axis_for_pair(a_id, b_id)
        if ax is None:
            return None
        return [(pa, pb) for _, pa, pb in ax.matched(a_id, b_id)]


def load_param_alignment(path: str | Path) -> ParamAlignment:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ParamAlignment.model_validate(raw)

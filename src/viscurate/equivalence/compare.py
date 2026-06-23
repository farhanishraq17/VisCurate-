"""The comparison primitive and the output provider (CLAUDE.md §3.5.1).

A *skill instance* ``σ = (skill, params)`` executed over the probe battery yields a
canonicalized output set ``O_σ``. The atomic comparison computes a **per-probe distance** at
three levels — pixel ``L∞``, LPIPS, DINO/CLIP cosine — then aggregates:

* **EXACT / PERCEPTUAL → worst probe (max).** Equivalence is universally quantified
  ("indistinguishable on *every* probe"); one diverging probe is a silent-merge bug in
  waiting (CLAUDE.md §3.5.1).
* **SEMANTIC_PRESERVING → high quantile (p90) + mean.** A distributional claim that tolerates
  a few hard probes; a pure ``max`` would make it unattainable.

The :class:`OutputProvider` protocol is the **text-blind boundary**: it yields
:class:`~viscurate.skills.model.ComparatorView` objects (no ``description``) and *outputs*,
never a :class:`~viscurate.skills.model.Skill`. :class:`BatteryEvaluator` is the trusted
implementation — it holds skills (with their ``fn``) and probe arrays internally and executes
them, but exposes only outputs, so a comparator cannot reach a description through the typed
interface.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from viscurate.equivalence.backends import PerceptualBackend, SemanticBackend, cosine_distance
from viscurate.skills.canonicalize import (
    Canonical,
    canonicalize,
    content_hash,
    max_abs_pixel_diff,
)
from viscurate.skills.model import ComparatorView, Image, Params, Skill

__all__ = [
    "Aggregate",
    "BatteryEvaluator",
    "OutputProvider",
    "OutputSet",
    "dino_distances",
    "lpips_distances",
    "pixel_distances",
]


@dataclass(frozen=True)
class OutputSet:
    """A skill instance's outputs over a battery: canonical view + raw arrays for composition.

    Only probes where the skill ran successfully are included; ``probe_ids`` is their order.
    """

    skill_id: str
    params_key: str
    probe_ids: tuple[str, ...]
    canon: Mapping[str, Canonical]
    raw: Mapping[str, Image]
    errors: Mapping[str, str]

    def common(self, other: OutputSet) -> tuple[str, ...]:
        """Probe ids where *both* sets ran (the only ones a comparison can use)."""
        rhs = set(other.probe_ids)
        return tuple(p for p in self.probe_ids if p in rhs)


@dataclass(frozen=True)
class Aggregate:
    """An aggregated distance with the probe that drove it (for worst-case evidence)."""

    value: float
    probe_id: str


def _params_key(params: Params | None) -> str:
    return json.dumps(params or {}, sort_keys=True, separators=(",", ":"))


class OutputProvider(Protocol):
    """The text-blind interface comparators are handed (CLAUDE.md §1.2)."""

    def comparator_view(self, skill_id: str) -> ComparatorView: ...

    def default_params(self, skill_id: str) -> Params: ...

    def identity_outputs(self) -> OutputSet: ...

    def outputs(
        self, skill_id: str, params: Params | None = None, *, seed: int | None = None
    ) -> OutputSet: ...

    def compose_outputs(
        self,
        outer_id: str,
        outer_params: Params | None,
        inner_id: str,
        inner_params: Params | None,
        *,
        seed: int | None = None,
    ) -> OutputSet: ...

    def param_grid(self, skill_id: str, *, max_points: int = 12) -> list[Params]: ...


class BatteryEvaluator:
    """Executes built-in (trusted) skills over a fixed battery, caching canonical outputs.

    Runs in-process (all built-ins are trusted) like the oracle freeze — the subprocess
    sandbox is for the trusted-gate/timeout contract, not for bulk sweeps. The same fixed
    ``seed`` is used for every execution so seeded-stochastic skills are compared at matched
    seeds (CLAUDE.md §1.4).
    """

    def __init__(
        self,
        skills: Sequence[Skill] | Mapping[str, Skill],
        battery: Sequence[tuple[str, Image]],
        *,
        seed: int = 0,
    ) -> None:
        self._skills: dict[str, Skill] = (
            dict(skills) if isinstance(skills, Mapping) else {s.id: s for s in skills}
        )
        self._battery: list[tuple[str, Image]] = list(battery)
        self._seed = seed
        self._cache: dict[tuple[str, str, int], OutputSet] = {}

    # -- text-blind interface -----------------------------------------------------
    def comparator_view(self, skill_id: str) -> ComparatorView:
        return self._skills[skill_id].comparator_view()

    def default_params(self, skill_id: str) -> Params:
        return self._skills[skill_id].params_schema.defaults()

    def identity_outputs(self) -> OutputSet:
        """The canonicalized raw battery inputs — the no-op reference for non-triviality."""
        key = ("__identity__", "", -1)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        canon: dict[str, Canonical] = {}
        raw: dict[str, Image] = {}
        ids: list[str] = []
        for probe_id, arr in self._battery:
            raw[probe_id] = arr
            canon[probe_id] = canonicalize(arr)
            ids.append(probe_id)
        result = OutputSet(
            skill_id="__identity__",
            params_key="",
            probe_ids=tuple(ids),
            canon=canon,
            raw=raw,
            errors={},
        )
        self._cache[key] = result
        return result

    def outputs(
        self, skill_id: str, params: Params | None = None, *, seed: int | None = None
    ) -> OutputSet:
        skill = self._skills[skill_id]
        validated = skill.params_schema.validate_params(params)
        use_seed = self._seed if seed is None else seed
        key = (skill_id, _params_key(validated), use_seed)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        ids: list[str] = []
        canon: dict[str, Canonical] = {}
        raw: dict[str, Image] = {}
        errors: dict[str, str] = {}
        for probe_id, arr in self._battery:
            try:
                out = skill.run(arr, validated, use_seed)
                raw[probe_id] = out
                canon[probe_id] = canonicalize(out)
                ids.append(probe_id)
            except Exception as exc:  # a skill legitimately failing on a probe is recorded
                errors[probe_id] = f"{type(exc).__name__}: {exc}"[:200]
        result = OutputSet(
            skill_id=skill_id,
            params_key=key[1],
            probe_ids=tuple(ids),
            canon=canon,
            raw=raw,
            errors=errors,
        )
        self._cache[key] = result
        return result

    def compose_outputs(
        self,
        outer_id: str,
        outer_params: Params | None,
        inner_id: str,
        inner_params: Params | None,
        *,
        seed: int | None = None,
    ) -> OutputSet:
        """Execute ``outer(inner(x))`` over the battery (the COMPLEMENTARY commutation test)."""
        outer = self._skills[outer_id]
        inner = self._skills[inner_id]
        v_outer = outer.params_schema.validate_params(outer_params)
        v_inner = inner.params_schema.validate_params(inner_params)
        use_seed = self._seed if seed is None else seed
        inner_out = self.outputs(inner_id, inner_params, seed=use_seed)
        ids: list[str] = []
        canon: dict[str, Canonical] = {}
        raw: dict[str, Image] = {}
        errors: dict[str, str] = dict(inner_out.errors)
        for probe_id in inner_out.probe_ids:
            try:
                out = outer.run(inner_out.raw[probe_id], v_outer, use_seed)
                raw[probe_id] = out
                canon[probe_id] = canonicalize(out)
                ids.append(probe_id)
            except Exception as exc:
                errors[probe_id] = f"{type(exc).__name__}: {exc}"[:200]
        key = f"{inner_id}:{_params_key(v_inner)}>>{outer_id}:{_params_key(v_outer)}"
        return OutputSet(
            skill_id=key,
            params_key="",
            probe_ids=tuple(ids),
            canon=canon,
            raw=raw,
            errors=errors,
        )

    def param_grid(self, skill_id: str, *, max_points: int = 12) -> list[Params]:
        """A coarse grid over a skill's numeric/enum params (subsumption fallback grid).

        Each numeric param is sampled at {min, mid, max, default}; enum/bool take all values;
        str takes its default. The Cartesian product is truncated deterministically to
        ``max_points`` to keep grids coarse (CLAUDE.md §3.5.4 — grid resolution is a knob).
        """
        schema = self._skills[skill_id].params_schema
        if not schema.params:
            return [{}]
        per_param: list[list[object]] = []
        names: list[str] = []
        for spec in schema.params:
            names.append(spec.name)
            per_param.append(_sample_param(spec))
        grid: list[Params] = [{}]
        for name, values in zip(names, per_param, strict=True):
            grid = [{**g, name: v} for g in grid for v in values]
            if len(grid) > max_points * 8:  # bound the intermediate blow-up
                grid = grid[: max_points * 8]
        # Deterministic truncation: keep the first `max_points` after a stable sort.
        grid.sort(key=lambda p: json.dumps(p, sort_keys=True))
        return grid[:max_points]


def _sample_param(spec: object) -> list[object]:
    from viscurate.skills.model import ParamSpec

    assert isinstance(spec, ParamSpec)
    if spec.type in ("bool",):
        return [True, False]
    if spec.type == "enum":
        return list(spec.choices or ())
    if spec.type == "str":
        return [spec.default]
    lo = spec.minimum
    hi = spec.maximum
    pts: list[object] = [spec.default]
    if lo is not None:
        pts.append(int(lo) if spec.type == "int" else float(lo))
    if hi is not None:
        pts.append(int(hi) if spec.type == "int" else float(hi))
    if lo is not None and hi is not None:
        mid = (lo + hi) / 2.0
        pts.append(int(round(mid)) if spec.type == "int" else float(mid))
    # de-dup preserving order
    seen: set[object] = set()
    out: list[object] = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# --------------------------------------------------------------------------------------------
# Per-probe distances + aggregation (the comparison primitive, §3.5.1).
# --------------------------------------------------------------------------------------------


def pixel_distances(a: OutputSet, b: OutputSet) -> dict[str, float]:
    """Worst-case-ready per-probe pixel ``L∞`` over the common probes (``inf`` on shape gate)."""
    return {p: max_abs_pixel_diff(a.canon[p], b.canon[p]) for p in a.common(b)}


def hashes_identical(a: OutputSet, b: OutputSet) -> bool:
    """True iff canonical content hashes match on every common probe (and there is overlap)."""
    common = a.common(b)
    if not common:
        return False
    return all(content_hash(a.canon[p]) == content_hash(b.canon[p]) for p in common)


def lpips_distances(a: OutputSet, b: OutputSet, backend: PerceptualBackend) -> dict[str, float]:
    return {p: backend.distance(a.canon[p].rgb, b.canon[p].rgb) for p in a.common(b)}


def dino_distances(a: OutputSet, b: OutputSet, backend: SemanticBackend) -> dict[str, float]:
    """Per-probe ``1 - cos`` in feature space; features batch-extracted once per set (§3.5.6)."""
    common = a.common(b)
    if not common:
        return {}
    fa = backend.features([a.canon[p].rgb for p in common])
    fb = backend.features([b.canon[p].rgb for p in common])
    return {p: cosine_distance(fa[i], fb[i]) for i, p in enumerate(common)}


def worst_case(dists: Mapping[str, float]) -> Aggregate:
    """The maximum distance and the probe that drove it (EXACT/PERCEPTUAL aggregation)."""
    if not dists:
        return Aggregate(value=float("inf"), probe_id="")
    probe_id = max(dists, key=lambda p: dists[p])
    return Aggregate(value=float(dists[probe_id]), probe_id=probe_id)


def quantile(dists: Mapping[str, float], q: float) -> float:
    """The ``q``-quantile of the distances (SEMANTIC aggregation uses p90)."""
    if not dists:
        return float("inf")
    return float(np.quantile(np.fromiter(dists.values(), dtype=np.float64), q))


def mean(dists: Mapping[str, float]) -> float:
    if not dists:
        return float("inf")
    return float(np.mean(np.fromiter(dists.values(), dtype=np.float64)))

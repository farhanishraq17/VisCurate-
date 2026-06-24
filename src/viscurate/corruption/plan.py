"""Deterministic corruption planning (CLAUDE.md §2.2 ρ-semantics, D3).

``plan_corruption`` is a pure function of ``(L0, ρ, composition c, seed, mode)`` that emits a
:class:`~viscurate.corruption.types.CorruptionLog`. The log is the seed-deterministic artifact:
``apply_corruption`` (a second pure function) replays it into a byte-identical ``L_ρ``. So the
exit criterion "same seed → byte-identical library" reduces to *this* being deterministic.

ρ = fraction of ``L0`` corrupted: ``K = round(ρ·N)`` distinct **sites** are chosen. In
single-defect mode each site gets exactly one defect, its type drawn from ``c`` via
**eligibility-aware Hamilton apportionment** — SUBSUMPTION needs a numeric parameter to bake,
PARAM_SCHEMA_BUG needs a parameter with an alternate value; the rest accept any skill. When a
heavily-weighted restricted type cannot be filled (small eligible pool at high ρ), the deficit
spills to the flexible types and the **realized** composition is recorded honestly — it is
never silently forced.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from viscurate.corruption.mutators import BUG_MUTATORS, DOMAINS
from viscurate.corruption.types import (
    IN_PLACE_TYPES,
    Composition,
    CorruptionEntry,
    CorruptionLog,
    CorruptionType,
)
from viscurate.rng import SeedManager, derive_seed
from viscurate.skills.model import ParamSpec, Skill

__all__ = ["plan_corruption"]

# Preferred outer ops for a DEAD_SKILL composition: each reliably changes a non-trivial input,
# so the composition is genuinely non-identity (CLAUDE.md §3.5.8 type 7).
_DEAD_OUTER_PREFERENCE: tuple[str, ...] = ("invert_v1", "sepia_tone_v1", "posterize_v1")

# Assignment order: most-restricted pools first, so they claim scarce skills before the
# flexible (any-skill) types consume the shared pool.
_FLEXIBLE_ORDER: tuple[CorruptionType, ...] = (
    CorruptionType.IMPLEMENTATION_BUG,
    CorruptionType.METADATA_MISLEAD,
    CorruptionType.DUPLICATE,
    CorruptionType.DOMAIN_SCOPED_BUG,
    CorruptionType.DEAD_SKILL,
)
_RESTRICTED_ORDER: tuple[CorruptionType, ...] = (
    CorruptionType.SUBSUMPTION,
    CorruptionType.PARAM_SCHEMA_BUG,
)


# --------------------------------------------------------------------------------------
# Parameter helpers — eligibility and the deterministic "wrong / specialized" value.
# --------------------------------------------------------------------------------------


def _has_alt(spec: ParamSpec) -> bool:
    """Whether this parameter admits a valid value distinct from its default."""
    if spec.type in ("int", "float"):
        return spec.minimum is None or spec.maximum is None or spec.maximum > spec.minimum
    if spec.type == "bool" or spec.type == "str":
        return True
    if spec.type == "enum":
        return spec.choices is not None and len(spec.choices) >= 2
    return False


def _numeric_with_alt(skill: Skill) -> list[ParamSpec]:
    return [p for p in skill.params_schema.params if p.type in ("int", "float") and _has_alt(p)]


def _params_with_alt(skill: Skill) -> list[ParamSpec]:
    return [p for p in skill.params_schema.params if _has_alt(p)]


def _alt_value(spec: ParamSpec, rng: np.random.Generator) -> float | int | bool | str:
    """A valid value distinct from ``spec.default`` (the baked specialization / wrong default)."""
    if spec.type == "bool":
        return not bool(spec.default)
    if spec.type == "enum":
        others = [c for c in (spec.choices or ()) if c != spec.default]
        chosen = others[int(rng.integers(len(others)))]
        # enum choices are typed Any; our schemas use str/int/bool/float choices.
        return chosen if isinstance(chosen, (bool, int, float, str)) else str(chosen)
    if spec.type == "str":
        return f"{spec.default}_alt"

    default = float(spec.default)
    lo, hi = spec.minimum, spec.maximum
    if lo is not None and hi is not None and hi > lo:
        candidates = [lo, hi, (lo + hi) / 2.0]
        distinct = [c for c in candidates if abs(c - default) > 1e-9] or [
            hi if default <= lo else lo
        ]
        value = float(distinct[int(rng.integers(len(distinct)))])
    elif lo is not None:
        value = default + 1.0
    elif hi is not None:
        value = default - 1.0
    else:
        value = default + 1.0

    if spec.type == "int":
        ivalue = int(round(value))
        if ivalue == int(round(default)):
            ivalue += 1
        if lo is not None:
            ivalue = max(ivalue, int(round(lo)))
        if hi is not None:
            ivalue = min(ivalue, int(round(hi)))
        return ivalue
    return value


# --------------------------------------------------------------------------------------
# Apportionment + assignment.
# --------------------------------------------------------------------------------------


def _hamilton(weights: dict[CorruptionType, float], k: int) -> dict[CorruptionType, int]:
    """Largest-remainder apportionment of ``k`` over normalized ``weights`` (deterministic)."""
    if k <= 0 or not weights:
        return {}
    quotas = {t: k * w for t, w in weights.items()}
    counts = {t: int(np.floor(q)) for t, q in quotas.items()}
    remainder = k - sum(counts.values())
    # Hand out the leftover to the largest fractional parts; tie-break by enum order.
    order = sorted(weights, key=lambda t: (-(quotas[t] - counts[t]), list(CorruptionType).index(t)))
    for t in order[:remainder]:
        counts[t] += 1
    return counts


def _shuffled(ids: Sequence[str], seed: int, *labels: str) -> list[str]:
    rng = SeedManager(seed).generator("corrupt", "shuffle", *labels)
    arr = list(ids)
    perm = rng.permutation(len(arr))
    return [arr[i] for i in perm]


def _assign_sites(
    skills: Sequence[Skill], k: int, weights: dict[CorruptionType, float], seed: int
) -> dict[CorruptionType, list[str]]:
    """Pick ``k`` distinct sites and a type for each; restricted pools claimed first."""
    by_id = {s.id: s for s in skills}
    all_ids = sorted(by_id)
    pools: dict[CorruptionType, list[str]] = {
        CorruptionType.SUBSUMPTION: sorted(s.id for s in skills if _numeric_with_alt(s)),
        CorruptionType.PARAM_SCHEMA_BUG: sorted(s.id for s in skills if _params_with_alt(s)),
    }
    for t in _FLEXIBLE_ORDER:
        pools[t] = all_ids

    # Normalize requested weights over types with a non-empty pool, then apportion + pool-cap.
    active = {t: w for t, w in weights.items() if w > 0 and pools.get(t)}
    total = sum(active.values())
    norm = {t: w / total for t, w in active.items()} if total > 0 else {}
    targets = {t: min(c, len(pools[t])) for t, c in _hamilton(norm, k).items()}

    used: set[str] = set()
    assigned: dict[CorruptionType, list[str]] = {t: [] for t in CorruptionType}
    for t in (*_RESTRICTED_ORDER, *_FLEXIBLE_ORDER):
        want = targets.get(t, 0)
        if want <= 0:
            continue
        avail = [i for i in _shuffled(pools[t], seed, "pool", t.value) if i not in used]
        take = avail[: min(want, len(avail))]
        assigned[t] = take
        used.update(take)

    # Absorb any deficit (restricted pools too small) into the flexible types — every skill is
    # eligible for those, so K distinct sites are always placeable when K ≤ N.
    placed = len(used)
    if placed < k:
        flex = [t for t in _FLEXIBLE_ORDER if weights.get(t, 0) > 0] or [
            CorruptionType.IMPLEMENTATION_BUG
        ]
        spill = [i for i in _shuffled(all_ids, seed, "overflow") if i not in used][: k - placed]
        for n, sid in enumerate(spill):
            assigned[flex[n % len(flex)]].append(sid)
            used.add(sid)
    return assigned


# --------------------------------------------------------------------------------------
# Per-defect entry construction.
# --------------------------------------------------------------------------------------


def _entry_for(t: CorruptionType, site: str, seed: int, by_id: dict[str, Skill]) -> CorruptionEntry:
    skill = by_id[site]
    rng = SeedManager(seed).generator("corrupt", "entry", t.value, site)

    if t is CorruptionType.IMPLEMENTATION_BUG:
        return CorruptionEntry(
            type=t, site_id=site, mutator=BUG_MUTATORS[int(rng.integers(len(BUG_MUTATORS)))]
        )

    if t is CorruptionType.DOMAIN_SCOPED_BUG:
        return CorruptionEntry(
            type=t,
            site_id=site,
            domain=DOMAINS[int(rng.integers(len(DOMAINS)))],
            mutator=BUG_MUTATORS[int(rng.integers(len(BUG_MUTATORS)))],
        )

    if t is CorruptionType.METADATA_MISLEAD:
        others = sorted(i for i in by_id if i != site)
        donor = by_id[others[int(rng.integers(len(others)))]]
        return CorruptionEntry(
            type=t,
            site_id=site,
            new_name=donor.name,
            new_description=donor.description,
            orig_name=skill.name,
            orig_description=skill.description,
        )

    if t is CorruptionType.PARAM_SCHEMA_BUG:
        candidates = _params_with_alt(skill)
        spec = candidates[int(rng.integers(len(candidates)))]
        return CorruptionEntry(
            type=t,
            site_id=site,
            param_name=spec.name,
            value=_alt_value(spec, rng),
            orig_value=spec.default,
        )

    if t is CorruptionType.DUPLICATE:
        variant = "perceptual" if derive_seed(seed, "dupvariant", site) & 1 else "exact"
        return CorruptionEntry(
            type=t, site_id=site, new_skill_id=f"{site}__dup_v1", variant=variant
        )

    if t is CorruptionType.SUBSUMPTION:
        candidates = _numeric_with_alt(skill)
        spec = candidates[int(rng.integers(len(candidates)))]
        return CorruptionEntry(
            type=t,
            site_id=site,
            new_skill_id=f"{site}__sub_v1",
            param_name=spec.name,
            value=_alt_value(spec, rng),
            orig_value=spec.default,
        )

    if t is CorruptionType.DEAD_SKILL:
        outer = next(
            (o for o in _DEAD_OUTER_PREFERENCE if o in by_id and o != site),
            next(i for i in sorted(by_id) if i != site),
        )
        return CorruptionEntry(
            type=t, site_id=site, new_skill_id=f"{site}__dead_v1", compose_with=outer
        )

    raise ValueError(f"unhandled corruption type {t!r}")  # pragma: no cover


def _mixed_extra(
    site: str, existing: CorruptionType, seed: int, skill: Skill
) -> CorruptionType | None:
    """In mixed mode, maybe pick a *second*, co-occurring in-place defect for ``site``.

    A site carries **at most one output-altering defect** (implementation / domain / schema)
    plus optionally METADATA_MISLEAD. This models the canonical realistic co-occurrence — "a
    skill that is both broken and mis-described" — while keeping every defect's invariant
    independently verifiable (stacking two output-altering bugs makes neither observable).
    """
    rng = SeedManager(seed).generator("corrupt", "mixed", site)
    if rng.random() >= 0.5:
        return None
    if existing is CorruptionType.METADATA_MISLEAD:
        options = [CorruptionType.IMPLEMENTATION_BUG, CorruptionType.DOMAIN_SCOPED_BUG]
        if _params_with_alt(skill):
            options.append(CorruptionType.PARAM_SCHEMA_BUG)
    else:  # `existing` is output-altering → only the output-invariant text defect may ride along
        options = [CorruptionType.METADATA_MISLEAD]
    options.sort(key=lambda t: list(CorruptionType).index(t))
    return options[int(rng.integers(len(options)))]


def plan_corruption(
    skills: Sequence[Skill],
    *,
    rho: float,
    composition: Composition,
    seed: int,
    mode: str = "single",
    version: str = "1.0.0",
) -> CorruptionLog:
    """Plan the defects for one ``(ρ, c, seed, mode)`` instance (deterministic)."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if mode not in ("single", "mixed"):
        raise ValueError(f"mode must be 'single' or 'mixed', got {mode!r}")

    n = len(skills)
    by_id = {s.id: s for s in skills}
    k = max(0, min(n, int(round(rho * n))))
    assigned = _assign_sites(skills, k, dict(composition.weights), seed)

    entries: list[CorruptionEntry] = []
    for t in CorruptionType:
        for site in assigned[t]:
            entries.append(_entry_for(t, site, seed, by_id))
            if mode == "mixed" and t in IN_PLACE_TYPES:
                extra = _mixed_extra(site, t, seed, by_id[site])
                if extra is not None:
                    entries.append(_entry_for(extra, site, seed, by_id))

    entries.sort(key=lambda e: (list(CorruptionType).index(e.type), e.site_id, e.new_skill_id))
    return CorruptionLog(
        version=version,
        rho=rho,
        composition=composition.name,
        composition_weights=dict(composition.weights),
        seed=seed,
        mode=mode,
        n_base=n,
        entries=tuple(entries),
    )

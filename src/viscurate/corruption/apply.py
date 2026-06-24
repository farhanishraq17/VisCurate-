"""Replay a corruption log into a corrupted library (CLAUDE.md §2.2, §2.3, §2.5).

``apply_corruption`` is a **pure function of ``(L0, log)``** — it reconstructs every corrupted
and added skill from the log's named factories, derives ``G_ρ`` by *appending* the log's
relation deltas to ``G0`` (never relabeling), and emits the ideal-action key. Determinism of
``plan_corruption`` plus purity here gives the exit criterion: same ``(ρ, c, seed, mode)`` →
byte-identical ``L_ρ`` and ``G_ρ``.

The added/mutated skills **masquerade as ordinary library skills**: their `provenance` stays
``"builtin"`` and the defect lives only in the internal-only `is_buggy` / `is_dead` labels (for
bugs / dead skills) or in the relational structure (for duplicates / subsumptions). The agent
must *discover* the defects from outputs and usage — it can never read them off a tag
(CLAUDE.md §1.2). The corruption log and ideal-action key are the only ground truth.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from viscurate.benchmark.ground_truth import GroundTruthGraph, GroundTruthSpec
from viscurate.corruption.mutators import (
    make_dead_fn,
    make_domain_bug_fn,
    make_exact_dup_fn,
    make_fixed_param_fn,
    make_impl_bug_fn,
    make_perceptual_dup_fn,
)
from viscurate.corruption.types import (
    CorruptionEntry,
    CorruptionLog,
    CorruptionType,
    IdealAction,
    IdealActionKind,
)
from viscurate.skills.model import ParamSpec, ParamsSchema, Skill, SkillMetadata
from viscurate.skills.registry import SkillRegistry

__all__ = ["CorruptedLibrary", "apply_corruption", "load_g0_spec"]


@dataclass(frozen=True)
class CorruptedLibrary:
    """The product of replaying a corruption log onto ``L0`` (CLAUDE.md §2.5 bundle)."""

    log: CorruptionLog
    registry: SkillRegistry
    g_rho: GroundTruthGraph
    g_rho_spec: GroundTruthSpec
    ideal_actions: tuple[IdealAction, ...]

    def n_added(self) -> int:
        return sum(1 for e in self.log.entries if e.is_add)


def load_g0_spec(path: str | Path) -> GroundTruthSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return GroundTruthSpec.model_validate(raw)


# --------------------------------------------------------------------------------------
# In-place mutations (fold all defects on one site into one final skill).
# --------------------------------------------------------------------------------------


def _corrupt_default(schema: ParamsSchema, param_name: str, value: object) -> ParamsSchema:
    """Return ``schema`` with one default replaced — re-validated (consistent but wrong)."""
    new_params = []
    for p in schema.params:
        if p.name == param_name:
            new_params.append(
                ParamSpec(
                    name=p.name,
                    type=p.type,
                    default=value,
                    minimum=p.minimum,
                    maximum=p.maximum,
                    choices=p.choices,
                    description=p.description,
                )
            )
        else:
            new_params.append(p)
    return ParamsSchema(params=tuple(new_params))


def _apply_in_place(base: Skill, entries: Sequence[CorruptionEntry]) -> Skill:
    fn = base.fn
    name, description = base.name, base.description
    schema = base.params_schema
    is_buggy = base.metadata.is_buggy
    # Deterministic fold order: enum order, so two entries on one site always compose identically.
    for e in sorted(entries, key=lambda x: list(CorruptionType).index(x.type)):
        if e.type is CorruptionType.IMPLEMENTATION_BUG:
            fn = make_impl_bug_fn(fn, e.mutator)
            is_buggy = True
        elif e.type is CorruptionType.DOMAIN_SCOPED_BUG:
            fn = make_domain_bug_fn(fn, e.domain, e.mutator)
            is_buggy = True
        elif e.type is CorruptionType.METADATA_MISLEAD:
            name, description = e.new_name, e.new_description
        elif e.type is CorruptionType.PARAM_SCHEMA_BUG:
            schema = _corrupt_default(schema, e.param_name, e.value)
    meta = base.metadata.model_copy(update={"is_buggy": is_buggy})
    return Skill(
        id=base.id,
        name=name,
        description=description,
        fn=fn,
        params_schema=schema,
        metadata=meta,
    )


# --------------------------------------------------------------------------------------
# Added skills (duplicate / subsumption / dead) + their G_ρ relation delta.
# --------------------------------------------------------------------------------------


def _build_added(e: CorruptionEntry, by_id: dict[str, Skill]) -> Skill:
    donor = by_id[e.site_id]
    fam = donor.metadata.family

    if e.type is CorruptionType.DUPLICATE:
        fn = (
            make_exact_dup_fn(donor.fn)
            if e.variant != "perceptual"
            else make_perceptual_dup_fn(donor.fn)
        )
        return Skill(
            id=e.new_skill_id,
            name=f"{donor.name} (copy)",
            description=donor.description,
            fn=fn,
            params_schema=donor.params_schema,
            metadata=donor.metadata.model_copy(update={"is_buggy": False, "is_dead": False}),
        )

    if e.type is CorruptionType.SUBSUMPTION:
        baked = donor.params_schema.defaults()
        baked[e.param_name] = e.value
        return Skill(
            id=e.new_skill_id,
            name=f"{donor.name} [{e.param_name}={e.value}]",
            description=f"Specialization of {donor.name} with {e.param_name} fixed at {e.value}.",
            fn=make_fixed_param_fn(donor.fn, baked),
            params_schema=ParamsSchema(),
            metadata=SkillMetadata(
                family=fam,
                seeded_stochastic=donor.metadata.seeded_stochastic,
                precision_sensitive=donor.metadata.precision_sensitive,
                platform_sensitive=donor.metadata.platform_sensitive,
            ),
        )

    if e.type is CorruptionType.DEAD_SKILL:
        outer = by_id[e.compose_with]
        fn = make_dead_fn(
            donor.fn, donor.params_schema.defaults(), outer.fn, outer.params_schema.defaults()
        )
        return Skill(
            id=e.new_skill_id,
            name=f"{outer.name} of {donor.name}",
            description=f"{outer.name} applied to the output of {donor.name} (composite).",
            fn=fn,
            params_schema=ParamsSchema(),
            metadata=SkillMetadata(
                family=fam,
                is_dead=True,
                seeded_stochastic=donor.metadata.seeded_stochastic
                or outer.metadata.seeded_stochastic,
            ),
        )

    raise ValueError(f"_build_added called on non-add type {e.type!r}")  # pragma: no cover


def _ideal_action(e: CorruptionEntry) -> IdealAction:
    t = e.type
    if t is CorruptionType.IMPLEMENTATION_BUG:
        return IdealAction(
            kind=IdealActionKind.REMOVE, type=t, primary=e.site_id, reason="implementation bug"
        )
    if t is CorruptionType.DOMAIN_SCOPED_BUG:
        return IdealAction(
            kind=IdealActionKind.MODIFY,
            type=t,
            primary=e.site_id,
            reason=f"domain-scoped bug on {e.domain}",
        )
    if t is CorruptionType.METADATA_MISLEAD:
        return IdealAction(
            kind=IdealActionKind.MODIFY, type=t, primary=e.site_id, reason="misleading metadata"
        )
    if t is CorruptionType.PARAM_SCHEMA_BUG:
        return IdealAction(
            kind=IdealActionKind.MODIFY,
            type=t,
            primary=e.site_id,
            reason=f"corrupted default for {e.param_name}",
        )
    if t is CorruptionType.DUPLICATE:
        return IdealAction(
            kind=IdealActionKind.MERGE,
            type=t,
            primary=e.new_skill_id,
            secondary=e.site_id,
            reason=f"{e.variant or 'exact'} duplicate of {e.site_id}",
        )
    if t is CorruptionType.SUBSUMPTION:
        return IdealAction(
            kind=IdealActionKind.PARAMETERIZE,
            type=t,
            primary=e.new_skill_id,
            secondary=e.site_id,
            reason=f"specialization subsumed by {e.site_id}",
        )
    if t is CorruptionType.DEAD_SKILL:
        return IdealAction(
            kind=IdealActionKind.REMOVE, type=t, primary=e.new_skill_id, reason="dead skill"
        )
    raise ValueError(f"unhandled corruption type {t!r}")  # pragma: no cover


def _augment_g0(
    g0: GroundTruthSpec,
    *,
    exact: list[tuple[str, str]],
    perceptual: list[tuple[str, str]],
    subsumption: list[tuple[str, str]],
) -> GroundTruthSpec:
    """Append injected relation deltas to ``G0`` (no existing label is touched)."""
    data = g0.model_dump(mode="python")
    data["exact"] = [tuple(p) for p in data["exact"]] + exact
    data["perceptual"] = [tuple(p) for p in data["perceptual"]] + perceptual
    data["subsumption"] = list(data["subsumption"]) + [
        {"spec": spec, "gen": gen} for spec, gen in subsumption
    ]
    return GroundTruthSpec.model_validate(data)


def apply_corruption(
    l0_skills: Sequence[Skill], log: CorruptionLog, g0: GroundTruthSpec
) -> CorruptedLibrary:
    """Replay ``log`` onto ``L0`` → corrupted registry, ``G_ρ``, and the ideal-action key."""
    by_id = {s.id: s for s in l0_skills}
    in_place: dict[str, list[CorruptionEntry]] = defaultdict(list)
    add_entries: list[CorruptionEntry] = []
    for e in log.entries:
        (add_entries if e.is_add else in_place[e.site_id]).append(e)

    registry = SkillRegistry()
    for s in l0_skills:  # preserve L0 order; mutate sites in place
        registry.register(_apply_in_place(s, in_place[s.id]) if s.id in in_place else s)

    # EXACT must stay transitively closed: collect dups per donor, emit the complete graph.
    exact_by_donor: dict[str, list[str]] = defaultdict(list)
    perceptual: list[tuple[str, str]] = []
    subsumption: list[tuple[str, str]] = []
    for e in sorted(add_entries, key=lambda x: x.new_skill_id):
        registry.register(_build_added(e, by_id))
        if e.type is CorruptionType.DUPLICATE:
            if e.variant == "perceptual":
                perceptual.append((e.site_id, e.new_skill_id))
            else:
                exact_by_donor[e.site_id].append(e.new_skill_id)
        elif e.type is CorruptionType.SUBSUMPTION:
            subsumption.append((e.new_skill_id, e.site_id))  # spec ⊑ gen

    exact: list[tuple[str, str]] = []
    for donor, dups in exact_by_donor.items():
        nodes = [donor, *dups]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                exact.append((nodes[i], nodes[j]))

    g_rho_spec = _augment_g0(g0, exact=exact, perceptual=perceptual, subsumption=subsumption)
    g_rho = GroundTruthGraph(g_rho_spec, valid_ids=set(registry.ids()))

    ideal_actions = tuple(
        _ideal_action(e)
        for e in sorted(
            log.entries,
            key=lambda x: (list(CorruptionType).index(x.type), x.site_id, x.new_skill_id),
        )
    )
    return CorruptedLibrary(
        log=log,
        registry=registry,
        g_rho=g_rho,
        g_rho_spec=g_rho_spec,
        ideal_actions=ideal_actions,
    )

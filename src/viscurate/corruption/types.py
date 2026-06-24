"""The corruption vocabulary and its serializable artifacts (CLAUDE.md §2.2, §2.4, §3.5.8).

Phase 5 turns the clean base library ``L0`` into a *family* of corrupted libraries ``L_ρ``
indexed by ``(ρ, composition c, seed, mode)`` — never a single static file. The
**corruption log is the canonical artifact**: it fully specifies, in JSON, how every
corrupted/added skill is derived from ``L0`` (which skill, which defect, which mutator, which
baked value …). :func:`viscurate.corruption.apply.apply_corruption` is a *pure* function of
``(L0, log)``, so ``same (ρ, c, seed, mode) → byte-identical L_ρ`` reduces to "the planner is
deterministic" (it is) — the fns themselves are reconstructed from the log via named factories,
never pickled.

The seven defect types (CLAUDE.md §2.2):

======  ==========================  ===============  ============================
 #       type                        verifier sees    G_ρ relation delta
======  ==========================  ===============  ============================
 1       IMPLEMENTATION_BUG          yes (≠ oracle)   none (DISTINCT-from-sibling)
 2       METADATA_MISLEAD            no (unchanged)   none
 3       DUPLICATE                   yes              EXACT / PERCEPTUAL
 4       SUBSUMPTION                 yes (directional) SUBSUMPTION
 5       PARAM_SCHEMA_BUG            partial          none
 6       DOMAIN_SCOPED_BUG           only if covered  none
 7       DEAD_SKILL                  no (utility)     none
======  ==========================  ===============  ============================

Types 1/2/5/6 mutate an existing ``L0`` skill **in place**; types 3/4/7 **add** a new skill
tied to a donor. Either way the donor/target is the corruption *site*; each ``L0`` skill is at
most one site (single-defect mode), so ``ρ`` = ``|sites| / |L0|`` is well-defined.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ADD_TYPES",
    "BUILTIN_COMPOSITIONS",
    "IN_PLACE_TYPES",
    "Composition",
    "CorruptionEntry",
    "CorruptionLog",
    "CorruptionManifest",
    "CorruptionType",
    "IdealAction",
    "IdealActionKind",
    "composition_by_name",
]


class CorruptionType(StrEnum):
    """The seven defect types injected at rate ρ (CLAUDE.md §2.2)."""

    IMPLEMENTATION_BUG = "implementation_bug"
    METADATA_MISLEAD = "metadata_mislead"
    DUPLICATE = "duplicate"
    SUBSUMPTION = "subsumption"
    PARAM_SCHEMA_BUG = "param_schema_bug"
    DOMAIN_SCOPED_BUG = "domain_scoped_bug"
    DEAD_SKILL = "dead_skill"


#: Types that add a new skill tied to a donor (the site stays clean in single-defect mode).
ADD_TYPES: frozenset[CorruptionType] = frozenset(
    {CorruptionType.DUPLICATE, CorruptionType.SUBSUMPTION, CorruptionType.DEAD_SKILL}
)
#: Types that mutate the site skill in place.
IN_PLACE_TYPES: frozenset[CorruptionType] = frozenset(
    {
        CorruptionType.IMPLEMENTATION_BUG,
        CorruptionType.METADATA_MISLEAD,
        CorruptionType.PARAM_SCHEMA_BUG,
        CorruptionType.DOMAIN_SCOPED_BUG,
    }
)


class IdealActionKind(StrEnum):
    """The curation action the dataset asserts is *correct* for a defect (CLAUDE.md §3.5.7).

    These are the structural/utility actions from the agent's action set
    (``merge / parameterize / modify / remove``) — the ideal-action key against which Phase-8
    curation is scored. ``KEEP`` is the correct action for an uncorrupted relation.
    """

    MERGE = "merge"
    PARAMETERIZE = "parameterize"
    MODIFY = "modify"
    REMOVE = "remove"
    KEEP = "keep"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CorruptionEntry(_Frozen):
    """One injected defect — a fully-serializable, replayable record (CLAUDE.md §2.5).

    A flat union of optional fields rather than a discriminated detail blob: every field is
    self-documenting and validated, and the empty defaults make a single-type entry readable.
    The fields used per type:

    * IMPLEMENTATION_BUG  : ``mutator``
    * DOMAIN_SCOPED_BUG   : ``mutator``, ``domain``
    * METADATA_MISLEAD    : ``new_name``, ``new_description``, ``orig_name``, ``orig_description``
    * PARAM_SCHEMA_BUG    : ``param_name``, ``value`` (the new, wrong default), ``orig_value``
    * DUPLICATE           : ``new_skill_id``, ``variant`` ("exact" | "perceptual")
    * SUBSUMPTION         : ``new_skill_id``, ``param_name``, ``value`` (the baked specialization)
    * DEAD_SKILL          : ``new_skill_id``, ``compose_with`` (the outer op id)
    """

    type: CorruptionType
    site_id: str
    new_skill_id: str = ""

    # implementation / domain bug
    mutator: str = ""
    domain: str = ""

    # duplicate
    variant: str = ""

    # subsumption (baked specialization) / param-schema (corrupted default). bool-before-int
    # in the union so a baked int (e.g. rotate angle 90) is not silently coerced to 90.0.
    param_name: str = ""
    value: bool | int | float | str | None = None
    orig_value: bool | int | float | str | None = None

    # dead skill
    compose_with: str = ""

    # metadata mislead
    new_name: str = ""
    new_description: str = ""
    orig_name: str = ""
    orig_description: str = ""

    @property
    def is_add(self) -> bool:
        return self.type in ADD_TYPES

    def affected_ids(self) -> tuple[str, ...]:
        """Skill ids this entry creates or mutates (for QA / accounting)."""
        return (self.new_skill_id,) if self.is_add else (self.site_id,)


class Composition(_Frozen):
    """A named weighting over the seven defect types — the composition vector ``c`` (CLAUDE.md D3).

    Weights need not sum to 1; they are normalized over the types whose eligible pool is
    non-empty at apportionment time. A zero weight excludes a type entirely.
    """

    name: str
    weights: dict[CorruptionType, float]


def _uniform() -> dict[CorruptionType, float]:
    return dict.fromkeys(CorruptionType, 1.0)


#: The ≥3 compositions the headline curve runs (CLAUDE.md D3 / §2.2): a uniform mix, a
#: redundancy-heavy mix (stresses dedup/merge), and a text-heavy mix (stresses the text
#: baselines / the agent's metadata reasoning).
BUILTIN_COMPOSITIONS: dict[str, Composition] = {
    "uniform": Composition(name="uniform", weights=_uniform()),
    "duplicate_heavy": Composition(
        name="duplicate_heavy",
        weights={
            CorruptionType.DUPLICATE: 4.0,
            CorruptionType.SUBSUMPTION: 2.0,
            CorruptionType.IMPLEMENTATION_BUG: 1.0,
            CorruptionType.METADATA_MISLEAD: 1.0,
            CorruptionType.PARAM_SCHEMA_BUG: 1.0,
            CorruptionType.DOMAIN_SCOPED_BUG: 1.0,
            CorruptionType.DEAD_SKILL: 1.0,
        },
    ),
    "metadata_heavy": Composition(
        name="metadata_heavy",
        weights={
            CorruptionType.METADATA_MISLEAD: 4.0,
            CorruptionType.DEAD_SKILL: 2.0,
            CorruptionType.IMPLEMENTATION_BUG: 1.0,
            CorruptionType.DUPLICATE: 1.0,
            CorruptionType.SUBSUMPTION: 1.0,
            CorruptionType.PARAM_SCHEMA_BUG: 1.0,
            CorruptionType.DOMAIN_SCOPED_BUG: 1.0,
        },
    ),
}


def composition_by_name(name: str) -> Composition:
    try:
        return BUILTIN_COMPOSITIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown composition {name!r}; known: {sorted(BUILTIN_COMPOSITIONS)}"
        ) from None


class IdealAction(_Frozen):
    """One entry of the **ideal-action key** (CLAUDE.md §3.5.7 relation→action map).

    The action the agent *should* take for this defect, with the skill(s) involved. This is
    the answer key Phase-8 curation is scored against — derived deterministically from the
    corruption log, never from any metric.
    """

    kind: IdealActionKind
    type: CorruptionType
    primary: str  # the skill to act on (the one to merge-away / parameterize-away / fix / drop)
    secondary: str = ""  # the surviving skill (merge target / generalizer), when applicable
    reason: str = ""


class CorruptionLog(_Frozen):
    """The canonical, replayable description of one corrupted library ``L_ρ`` (CLAUDE.md §2.5)."""

    version: str = "1.0.0"
    rho: float
    composition: str
    composition_weights: dict[CorruptionType, float]
    seed: int
    mode: str  # "single" | "mixed"
    n_base: int
    entries: tuple[CorruptionEntry, ...]

    def realized_counts(self) -> dict[str, int]:
        out = {t.value: 0 for t in CorruptionType}
        for e in self.entries:
            out[e.type.value] += 1
        return out

    def sites(self) -> set[str]:
        return {e.site_id for e in self.entries}


class CorruptionManifest(_Frozen):
    """Reproducibility manifest stamped beside each ``(ρ, c, seed, mode)`` instance (§5)."""

    generator_version: str
    canon_version: str
    l0_specs_sha256: str
    g0_sha256: str
    rho: float
    composition: str
    seed: int
    mode: str
    n_base: int
    n_sites: int
    n_added: int
    n_skills_lrho: int
    realized_counts: dict[str, int]

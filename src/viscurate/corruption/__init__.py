"""Phase 5 — the corruption generator (CLAUDE.md §2.2, §2.4, Phase-5 deliverable).

Turns the clean base library ``L0`` into a *family* of corrupted libraries ``L_ρ`` indexed by
``(ρ, composition c, seed, mode)`` by injecting the seven defect types at a controlled rate.
The pipeline is two pure functions around a serializable log:

    plan_corruption(L0, ρ, c, seed, mode) -> CorruptionLog      # deterministic
    apply_corruption(L0, log, G0)         -> CorruptedLibrary   # pure replay → L_ρ, G_ρ, key

``run_qa`` confirms each injected defect took effect (the asymmetry that justifies the
verifier/agent split), and ``generate_grid`` sweeps the whole ρ-series and writes the
ground-truth bundle per instance.
"""

from __future__ import annotations

from viscurate.corruption.apply import CorruptedLibrary, apply_corruption, load_g0_spec
from viscurate.corruption.grid import (
    CorruptionGridConfig,
    generate_grid,
    generate_instance,
    instance_name,
)
from viscurate.corruption.plan import plan_corruption
from viscurate.corruption.qa import QAReport, QAResult, run_qa
from viscurate.corruption.types import (
    BUILTIN_COMPOSITIONS,
    Composition,
    CorruptionEntry,
    CorruptionLog,
    CorruptionManifest,
    CorruptionType,
    IdealAction,
    IdealActionKind,
    composition_by_name,
)

__all__ = [
    "BUILTIN_COMPOSITIONS",
    "Composition",
    "CorruptedLibrary",
    "CorruptionEntry",
    "CorruptionGridConfig",
    "CorruptionLog",
    "CorruptionManifest",
    "CorruptionType",
    "IdealAction",
    "IdealActionKind",
    "QAReport",
    "QAResult",
    "apply_corruption",
    "composition_by_name",
    "generate_grid",
    "generate_instance",
    "instance_name",
    "load_g0_spec",
    "plan_corruption",
    "run_qa",
]

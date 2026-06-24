"""The ``(ρ, composition c, seed, mode)`` grid driver (CLAUDE.md D2/D3, Phase-5 deliverable).

Generates the *family* of corrupted libraries — the graded ρ-series ρ ∈ {10%,…,100%} × ≥3
compositions × ≥5 seeds (+ a mixed/realistic mode) — and writes, per instance, the full
ground-truth bundle (CLAUDE.md §2.5): the corruption log, the corrupted library, ``G_ρ``, the
ideal-action key, a QA report (when a probe battery is supplied), and a reproducibility
manifest. Each instance is reproducible from its manifest alone.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from viscurate.benchmark.ground_truth import GroundTruthSpec
from viscurate.corruption.apply import CorruptedLibrary, apply_corruption
from viscurate.corruption.plan import plan_corruption
from viscurate.corruption.qa import QAReport, run_qa
from viscurate.corruption.types import (
    CorruptionManifest,
    composition_by_name,
)
from viscurate.logging import get_logger
from viscurate.skills.canonicalize import CANON_VERSION
from viscurate.skills.model import Image, Skill
from viscurate.skills.registry import SkillRegistry

__all__ = [
    "GENERATOR_VERSION",
    "CorruptionGridConfig",
    "generate_grid",
    "generate_instance",
    "instance_name",
]

GENERATOR_VERSION = "1.0.0"

_Probe = tuple[str, Image]

_DEFAULT_RHO = tuple(round(0.1 * i, 2) for i in range(1, 11))  # 0.1 … 1.0
_DEFAULT_COMPOSITIONS = ("uniform", "duplicate_heavy", "metadata_heavy")
_DEFAULT_SEEDS = (1234, 2345, 3456, 4567, 5678)  # ≥5 seeds for error bars (CLAUDE.md D3)


class CorruptionGridConfig(BaseModel):
    """The grid to sweep. Compositions are referenced by name (see ``BUILTIN_COMPOSITIONS``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rho_values: tuple[float, ...] = _DEFAULT_RHO
    compositions: tuple[str, ...] = _DEFAULT_COMPOSITIONS
    seeds: tuple[int, ...] = _DEFAULT_SEEDS
    modes: tuple[str, ...] = ("single",)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CorruptionGridConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)


def instance_name(rho: float, composition: str, seed: int, mode: str) -> str:
    return f"rho{int(round(rho * 100)):03d}_{composition}_seed{seed}_{mode}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _l0_specs_sha256(l0_skills: Sequence[Skill]) -> str:
    reg = SkillRegistry()
    for s in l0_skills:
        reg.register(s)
    return _sha256_text(reg.to_json())


def generate_instance(
    l0_skills: Sequence[Skill],
    g0_spec: GroundTruthSpec,
    *,
    rho: float,
    composition: str,
    seed: int,
    mode: str,
    out_dir: str | Path | None = None,
    probes: Sequence[_Probe] | None = None,
    l0_sha: str | None = None,
    g0_sha: str | None = None,
) -> tuple[CorruptedLibrary, QAReport | None, CorruptionManifest]:
    """Plan → apply → (optional QA) → write artifacts for one instance; return the bundle."""
    comp = composition_by_name(composition)
    log = plan_corruption(l0_skills, rho=rho, composition=comp, seed=seed, mode=mode)
    lib = apply_corruption(l0_skills, log, g0_spec)
    qa = run_qa(l0_skills, lib, probes) if probes else None

    manifest = CorruptionManifest(
        generator_version=GENERATOR_VERSION,
        canon_version=CANON_VERSION,
        l0_specs_sha256=l0_sha or _l0_specs_sha256(l0_skills),
        g0_sha256=g0_sha or _sha256_text(g0_spec.model_dump_json()),
        rho=rho,
        composition=composition,
        seed=seed,
        mode=mode,
        n_base=log.n_base,
        n_sites=len(log.sites()),
        n_added=lib.n_added(),
        n_skills_lrho=len(lib.registry),
        realized_counts=log.realized_counts(),
    )

    if out_dir is not None:
        dest = Path(out_dir) / instance_name(rho, composition, seed, mode)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "corruption_log.json").write_text(log.model_dump_json(indent=2), encoding="utf-8")
        (dest / "library.json").write_text(lib.registry.to_json(), encoding="utf-8")
        (dest / "g_rho.json").write_text(lib.g_rho_spec.model_dump_json(indent=2), encoding="utf-8")
        (dest / "ideal_actions.json").write_text(
            json.dumps([a.model_dump(mode="json") for a in lib.ideal_actions], indent=2),
            encoding="utf-8",
        )
        if qa is not None:
            (dest / "qa_report.json").write_text(qa.model_dump_json(indent=2), encoding="utf-8")
        (dest / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    return lib, qa, manifest


def generate_grid(
    cfg: CorruptionGridConfig,
    l0_skills: Sequence[Skill],
    g0_spec: GroundTruthSpec,
    out_dir: str | Path,
    *,
    probes: Sequence[_Probe] | None = None,
) -> list[CorruptionManifest]:
    """Generate every ``(ρ, c, seed, mode)`` instance in the grid; return their manifests."""
    log = get_logger("corruption.grid")
    l0_sha = _l0_specs_sha256(l0_skills)
    g0_sha = _sha256_text(g0_spec.model_dump_json())
    manifests: list[CorruptionManifest] = []
    for mode in cfg.modes:
        for composition in cfg.compositions:
            for rho in cfg.rho_values:
                for seed in cfg.seeds:
                    _lib, qa, manifest = generate_instance(
                        l0_skills,
                        g0_spec,
                        rho=rho,
                        composition=composition,
                        seed=seed,
                        mode=mode,
                        out_dir=out_dir,
                        probes=probes,
                        l0_sha=l0_sha,
                        g0_sha=g0_sha,
                    )
                    manifests.append(manifest)
                    log.info(
                        "corruption_instance",
                        name=instance_name(rho, composition, seed, mode),
                        sites=manifest.n_sites,
                        added=manifest.n_added,
                        qa=qa.counts() if qa is not None else None,
                    )
    return manifests

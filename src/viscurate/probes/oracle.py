"""The frozen reference oracle (CLAUDE.md §2.1).

Executes every clean ``L0`` skill over the whole probe battery and records, per
``(skill, probe)``, the canonicalized output's content hash (and shape / mask flag), or an
``error`` status when a skill legitimately cannot run on a probe (e.g. a 5×5 blur on a 1×1
degenerate image — itself a recorded baseline behaviour). The oracle is what later **confirms
corruption took effect** and **scores** the verifier/agent; it is *never* used to assign
relation labels.

Skills run **in-process** (all 100 are trusted built-ins) for speed — the subprocess
executor is for the trusted-gate/timeout contract, not for an 18k-execution sweep. Every
execution uses one fixed ``oracle_seed`` so the seeded-stochastic skills are reproducible.
Output arrays are *not* stored by default: they regenerate exactly from the deterministic
skills + probes, so the hash table is the reproducibility-critical artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from viscurate.equivalence.param_alignment import ParamAlignment
from viscurate.logging import get_logger
from viscurate.probes.build import load_probe
from viscurate.probes.manifest import ProbeManifest
from viscurate.skills.canonicalize import CANON_VERSION, canonicalize, content_hash
from viscurate.skills.model import Params
from viscurate.skills.registry import SkillRegistry

__all__ = [
    "ORACLE_VERSION",
    "OracleEntry",
    "OracleManifest",
    "freeze_oracle",
    "freeze_sweep_oracle",
    "verify_oracle",
]

ORACLE_VERSION = "1.0.0"


class OracleEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: str
    probe_id: str
    params_key: str = "{}"
    status: str  # "ok" | "error" | "nondeterministic"
    output_sha256: str = ""
    height: int = 0
    width: int = 0
    is_binary_mask: bool = False
    error: str = ""


class OracleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    oracle_version: str = ORACLE_VERSION
    canon_version: str
    battery_seed: int
    oracle_seed: int
    artifact_kind: str = "default_oracle"
    alignment_version: str = ""
    entries: tuple[OracleEntry, ...]

    def key(self, skill_id: str, probe_id: str, params_key: str = "{}") -> OracleEntry | None:
        for e in self.entries:  # linear is fine for verify on a subset; index if needed at scale
            if e.skill_id == skill_id and e.probe_id == probe_id and e.params_key == params_key:
                return e
        return None

    def status_counts(self) -> dict[str, int]:
        out = {"ok": 0, "error": 0, "nondeterministic": 0}
        for e in self.entries:
            out[e.status] = out.get(e.status, 0) + 1
        return out


def _run_one(
    registry: SkillRegistry,
    skill_id: str,
    image: object,
    oracle_seed: int,
    *,
    params: Params | None = None,
    check_determinism: bool = True,
) -> OracleEntry:
    """Run one (skill, probe). A reference is frozen only if the output is stable.

    With ``check_determinism`` the skill is run twice; if the canonical hashes differ the pair
    is recorded ``nondeterministic`` (no stable hash) rather than freezing an unstable value —
    e.g. ``inpaint`` on a 1-row degenerate probe. Such pairs are excluded from later
    corruption comparison.
    """
    skill = registry.get(skill_id)
    params_key = _params_key(params)
    try:
        canon = canonicalize(skill.run(image, params, seed=oracle_seed))  # type: ignore[arg-type]
        sha = content_hash(canon)
        status = "ok"
        if check_determinism:
            sha2 = content_hash(canonicalize(skill.run(image, params, seed=oracle_seed)))  # type: ignore[arg-type]
            if sha2 != sha:
                status = "nondeterministic"
        return OracleEntry(
            skill_id=skill_id,
            probe_id="",  # filled by caller
            params_key=params_key,
            status=status,
            output_sha256=sha if status == "ok" else "",
            height=canon.height,
            width=canon.width,
            is_binary_mask=canon.is_binary_mask,
        )
    except Exception as exc:
        return OracleEntry(
            skill_id=skill_id,
            probe_id="",
            params_key=params_key,
            status="error",
            error=f"{type(exc).__name__}: {exc}"[:200],
        )


def _params_key(params: Params | None) -> str:
    return json.dumps(params or {}, sort_keys=True, separators=(",", ":"))


def freeze_oracle(
    manifest: ProbeManifest,
    probe_dir: str | Path,
    registry: SkillRegistry,
    *,
    oracle_seed: int = 0,
    skill_ids: list[str] | None = None,
) -> OracleManifest:
    """Run every (skill, probe) once and record the output hash or error status."""
    log = get_logger("probes.oracle")
    probe_dir = Path(probe_dir)
    ids = skill_ids if skill_ids is not None else registry.ids()
    entries: list[OracleEntry] = []
    for entry in manifest.entries:
        image = load_probe(probe_dir, entry.probe_id)
        for sid in ids:
            e = _run_one(registry, sid, image, oracle_seed)
            entries.append(e.model_copy(update={"probe_id": entry.probe_id}))
    om = OracleManifest(
        canon_version=CANON_VERSION,
        battery_seed=manifest.seed,
        oracle_seed=oracle_seed,
        entries=tuple(entries),
    )
    log.info("oracle_frozen", pairs=len(entries), status=om.status_counts())
    return om


def freeze_sweep_oracle(
    manifest: ProbeManifest,
    probe_dir: str | Path,
    registry: SkillRegistry,
    alignment: ParamAlignment,
    *,
    oracle_seed: int = 0,
    skill_ids: list[str] | None = None,
) -> OracleManifest:
    """Freeze parameter-sweep outputs keyed by ``(skill, params_key, probe)``."""
    log = get_logger("probes.sweep_oracle")
    probe_dir = Path(probe_dir)
    ids = skill_ids if skill_ids is not None else registry.ids()
    entries: list[OracleEntry] = []
    for entry in manifest.entries:
        image = load_probe(probe_dir, entry.probe_id)
        for sid in ids:
            bindings = alignment.grid_for(sid)
            sub_grid = alignment.subsumption_grid(sid)
            if sub_grid is not None:
                bindings.extend(sub_grid)
            seen: set[str] = set()
            for params in bindings:
                key = _params_key(params)
                if key in seen:
                    continue
                seen.add(key)
                e = _run_one(registry, sid, image, oracle_seed, params=params)
                entries.append(e.model_copy(update={"probe_id": entry.probe_id}))
    om = OracleManifest(
        canon_version=CANON_VERSION,
        battery_seed=manifest.seed,
        oracle_seed=oracle_seed,
        artifact_kind="sweep_oracle",
        alignment_version=alignment.version,
        entries=tuple(entries),
    )
    log.info("sweep_oracle_frozen", pairs=len(entries), status=om.status_counts())
    return om


def verify_oracle(
    oracle: OracleManifest,
    manifest: ProbeManifest,
    probe_dir: str | Path,
    registry: SkillRegistry,
) -> list[tuple[str, str]]:
    """Re-run the stable pairs and return any `(skill_id, probe_id)` whose hash/status diverged.

    ``nondeterministic`` pairs are skipped (they have no stable reference to verify against).
    """
    probe_dir = Path(probe_dir)
    cache: dict[str, object] = {}
    mismatches: list[tuple[str, str]] = []
    for e in oracle.entries:
        if e.status == "nondeterministic":
            continue
        if e.probe_id not in cache:
            cache[e.probe_id] = load_probe(probe_dir, e.probe_id)
        again = _run_one(
            registry,
            e.skill_id,
            cache[e.probe_id],
            oracle.oracle_seed,
            params=json.loads(e.params_key),
            check_determinism=False,
        )
        if again.status != e.status or again.output_sha256 != e.output_sha256:
            mismatches.append((e.skill_id, e.probe_id))
    return mismatches

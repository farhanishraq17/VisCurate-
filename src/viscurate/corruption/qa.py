"""Per-type QA assertions (CLAUDE.md §2.2 — every injector has a confirming assertion).

A defect is only useful if it actually took effect, and the *kind* of effect is itself part
of the dataset's argument: IMPLEMENTATION_BUG and DOMAIN_SCOPED_BUG must **diverge** from the
clean output, while METADATA_MISLEAD and PARAM_SCHEMA_BUG must leave outputs **unchanged** (the
opposite assertion — the property that forces the agent's distinct, text/usage-based job).

These checks compare the corrupted library's skills to freshly-executed clean ``L0`` skills over
a probe battery, at **matched seeds** (so seeded-stochastic skills compare apples-to-apples).
The frozen oracle is an equivalent reference; re-running the deterministic clean skills here
keeps QA self-contained (no oracle file needed) and is what the exit criterion checks.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from viscurate.corruption.apply import CorruptedLibrary
from viscurate.corruption.mutators import matches_domain
from viscurate.corruption.types import CorruptionEntry, CorruptionType
from viscurate.skills.canonicalize import canonicalize, content_hash, max_abs_pixel_diff
from viscurate.skills.model import Image, Params, Skill

__all__ = ["QAReport", "QAResult", "run_qa"]

QA_SEED = 0
#: Worst-case L∞ (in [0,1]) a PERCEPTUAL duplicate may differ from its donor — the ≤1-LSB
#: dither stays well under this. Used only as a QA sanity bound, not the calibrated τ.
PERCEPTUAL_BOUND = 2.0 / 255.0

_Probe = tuple[str, Image]


class QAResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    type: CorruptionType
    site_id: str
    target_id: str
    status: str  # "pass" | "fail" | "skip"
    reason: str
    n_probes: int = 0


class QAReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[QAResult, ...]

    def counts(self) -> dict[str, int]:
        out = {"pass": 0, "fail": 0, "skip": 0}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    @property
    def all_passed(self) -> bool:
        """True if no assertion failed (skips are allowed — e.g. a domain absent from battery)."""
        return all(r.status != "fail" for r in self.results)


def _hash(skill: Skill, img: Image, params: Params | None, seed: int) -> str | None:
    try:
        return content_hash(canonicalize(skill.run(img, params, seed)))
    except Exception:
        return None


def _pixel_diff(
    a: Skill, b: Skill, img: Image, pa: Params | None, pb: Params | None, seed: int
) -> float | None:
    try:
        ca = canonicalize(a.run(img, pa, seed))
        cb = canonicalize(b.run(img, pb, seed))
    except Exception:
        return None
    return max_abs_pixel_diff(ca, cb)


def _check(
    e: CorruptionEntry,
    idx: int,
    clean: dict[str, Skill],
    lib: CorruptedLibrary,
    probes: Sequence[_Probe],
    cooccur: dict[str, set[CorruptionType]],
) -> QAResult:
    t = e.type
    reg = lib.registry
    # In mixed mode a site can carry several in-place defects. An output-invariant assertion
    # (metadata / param-schema) is only checkable in isolation; when an output-altering defect
    # co-occurs we assert the part that still holds (text/schema changed).
    others = cooccur.get(e.site_id, set()) - {t}
    _fn_bugs = {CorruptionType.IMPLEMENTATION_BUG, CorruptionType.DOMAIN_SCOPED_BUG}
    fn_altered = bool(others & _fn_bugs)

    def result(target: str, status: str, reason: str, n: int) -> QAResult:
        return QAResult(
            index=idx,
            type=t,
            site_id=e.site_id,
            target_id=target,
            status=status,
            reason=reason,
            n_probes=n,
        )

    if t is CorruptionType.IMPLEMENTATION_BUG:
        corrupt = reg.get(e.site_id)
        base = clean[e.site_id]
        diff = ran = 0
        for _pid, img in probes:
            hc, hb = _hash(corrupt, img, None, QA_SEED), _hash(base, img, None, QA_SEED)
            if hc is None or hb is None:
                continue
            ran += 1
            diff += hc != hb
        if ran == 0:
            return result(e.site_id, "skip", "no probe ran", 0)
        ok = diff > 0
        return result(e.site_id, "pass" if ok else "fail", f"diverged on {diff}/{ran} probes", ran)

    if t is CorruptionType.DOMAIN_SCOPED_BUG:
        corrupt, base = reg.get(e.site_id), clean[e.site_id]
        off_changed = on_changed = on_n = off_n = 0
        for _pid, img in probes:
            hc, hb = _hash(corrupt, img, None, QA_SEED), _hash(base, img, None, QA_SEED)
            if hc is None or hb is None:
                continue
            if matches_domain(img, e.domain):
                on_n += 1
                on_changed += hc != hb
            else:
                off_n += 1
                off_changed += hc != hb
        if on_n == 0:
            return result(e.site_id, "skip", f"battery has no {e.domain} probe", off_n)
        if off_changed:
            return result(
                e.site_id, "fail", f"fired off-domain on {off_changed}/{off_n}", on_n + off_n
            )
        ok = on_changed > 0
        return result(
            e.site_id,
            "pass" if ok else "fail",
            f"{e.domain}: changed {on_changed}/{on_n}",
            on_n + off_n,
        )

    if t is CorruptionType.METADATA_MISLEAD:
        corrupt, base = reg.get(e.site_id), clean[e.site_id]
        text_changed = corrupt.name != base.name or corrupt.description != base.description
        if others:  # a co-occurring defect changes the default output → check text only
            return result(
                e.site_id,
                "pass" if text_changed else "fail",
                f"text_changed={text_changed} (co-occurs with {sorted(o.value for o in others)})",
                0,
            )
        unchanged = ran = 0
        for _pid, img in probes:
            hc, hb = _hash(corrupt, img, None, QA_SEED), _hash(base, img, None, QA_SEED)
            if hc is None or hb is None:
                continue
            ran += 1
            unchanged += hc == hb
        if ran == 0:
            return result(e.site_id, "skip", "no probe ran", 0)
        ok = text_changed and unchanged == ran
        return result(
            e.site_id,
            "pass" if ok else "fail",
            f"text_changed={text_changed}, outputs unchanged {unchanged}/{ran}",
            ran,
        )

    if t is CorruptionType.PARAM_SCHEMA_BUG:
        corrupt, base = reg.get(e.site_id), clean[e.site_id]
        schema_changed = corrupt.params_schema.defaults() != base.params_schema.defaults()
        if fn_altered:  # a co-occurring fn bug breaks fn-identity → assert the schema delta only
            return result(
                e.site_id,
                "pass" if schema_changed else "fail",
                f"schema_changed={schema_changed} (fn altered by co-defect)",
                0,
            )
        clean_defaults = base.params_schema.defaults()  # fn identity: run both at the SAME params
        identical = ran = 0
        for _pid, img in probes:
            hc = _hash(corrupt, img, clean_defaults, QA_SEED)
            hb = _hash(base, img, clean_defaults, QA_SEED)
            if hc is None or hb is None:
                continue
            ran += 1
            identical += hc == hb
        if ran == 0:
            return result(e.site_id, "skip", "no probe ran", 0)
        ok = schema_changed and identical == ran
        return result(
            e.site_id,
            "pass" if ok else "fail",
            f"schema_changed={schema_changed}, fn identical {identical}/{ran}",
            ran,
        )

    if t is CorruptionType.DUPLICATE:
        dup, donor = reg.get(e.new_skill_id), clean[e.site_id]
        exact_n = ran = within = 0
        for _pid, img in probes:
            hc, hb = _hash(dup, img, None, QA_SEED), _hash(donor, img, None, QA_SEED)
            if hc is None or hb is None:
                continue
            ran += 1
            exact_n += hc == hb
            d = _pixel_diff(dup, donor, img, None, None, QA_SEED)
            within += d is not None and d <= PERCEPTUAL_BOUND
        if ran == 0:
            return result(e.new_skill_id, "skip", "no probe ran", 0)
        if e.variant == "perceptual":
            ok = exact_n < ran and within == ran  # close everywhere, but not byte-identical
            return result(
                e.new_skill_id,
                "pass" if ok else "fail",
                f"perceptual: exact {exact_n}/{ran}, within {within}/{ran}",
                ran,
            )
        ok = exact_n == ran
        return result(
            e.new_skill_id, "pass" if ok else "fail", f"exact: {exact_n}/{ran} identical", ran
        )

    if t is CorruptionType.SUBSUMPTION:
        spec, donor = reg.get(e.new_skill_id), clean[e.site_id]
        baked = donor.params_schema.defaults()
        baked[e.param_name] = e.value
        equal = ran = 0
        for _pid, img in probes:
            hs = _hash(spec, img, None, QA_SEED)
            hd = _hash(donor, img, baked, QA_SEED)
            if hs is None or hd is None:
                continue
            ran += 1
            equal += hs == hd
        if ran == 0:
            return result(e.new_skill_id, "skip", "no probe ran", 0)
        ok = equal == ran  # spec reproduces donor@baked exactly → subsumption holds
        return result(
            e.new_skill_id,
            "pass" if ok else "fail",
            f"reproduces donor@{e.param_name}={e.value} on {equal}/{ran}",
            ran,
        )

    if t is CorruptionType.DEAD_SKILL:
        dead, donor = reg.get(e.new_skill_id), clean[e.site_id]
        is_dead = dead.metadata.is_dead
        diff_donor = diff_identity = ran = 0
        for _pid, img in probes:
            hd = _hash(dead, img, None, QA_SEED)
            hb = _hash(donor, img, None, QA_SEED)
            if hd is None:
                continue
            ran += 1
            if hb is not None:
                diff_donor += hd != hb
            diff_identity += hd != content_hash(canonicalize(img))
        if ran == 0:
            return result(e.new_skill_id, "skip", "dead skill never ran", 0)
        ok = is_dead and diff_donor > 0 and diff_identity > 0
        return result(
            e.new_skill_id,
            "pass" if ok else "fail",
            f"is_dead={is_dead}, ≠donor {diff_donor}/{ran}, ≠identity {diff_identity}/{ran}",
            ran,
        )

    return result(e.site_id, "skip", f"no QA for {t}", 0)  # pragma: no cover


def run_qa(l0_skills: Sequence[Skill], lib: CorruptedLibrary, probes: Sequence[_Probe]) -> QAReport:
    """Run every entry's confirming assertion over ``probes`` and collect the verdicts."""
    clean = {s.id: s for s in l0_skills}
    cooccur: dict[str, set[CorruptionType]] = {}
    for e in lib.log.entries:
        if not e.is_add:  # only in-place defects share a site and can interact
            cooccur.setdefault(e.site_id, set()).add(e.type)
    results = [_check(e, i, clean, lib, probes, cooccur) for i, e in enumerate(lib.log.entries)]
    return QAReport(results=tuple(results))

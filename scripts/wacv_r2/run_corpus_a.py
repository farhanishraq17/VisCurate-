#!/usr/bin/env python
"""A1 Corpus A — execute the cross-library functional-core audit.

Pipeline, in the order A1/01 §5 prescribes:

  1. Load the frozen alignment map + published-row join, hash both.
  2. Build one repo-native ``Skill`` per implementation (adapters in ``corpus_a_lib``).
  3. FREEZE the pair list: write it with a sha256 BEFORE the first execution. ``--freeze-only``
     stops here so the artifact can be committed first.
  4. SELF-PAIR SANITY GATE: every implementation against itself must certify EXACT. A failure
     means the adapter or the callable is non-deterministic, so no divergence below is
     interpretable; the gate result is reported either way and never silently skipped.
  5. Classify every within-family pair through the UNMODIFIED ``taxonomy.classify`` at the
     thresholds carried over from the synthetic Stage-1 run. NOTHING is calibrated here
     (A1/01 §5 step 4) — ``--thresholds`` must name an already-calibrated file, and its
     provenance stamp is copied into the manifest.
  6. Emit the claim-vs-execution matrix over the joined published rows, every rate carrying raw
     counts and an exact Clopper-Pearson interval.

Alignment failures and behavioral non-equivalence are kept in separate columns throughout
(A1/01 §4.1 rule 5); ``variants`` rows are reported as alignment SENSITIVITY, not as findings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_a_lib import ADAPTERS, ALBUCORE_SHIM, CALL_GLOBALS

from viscurate.config import ThresholdConfig
from viscurate.equivalence.compare import BatteryEvaluator
from viscurate.equivalence.param_alignment import ParamAlignment
from viscurate.equivalence.relations import Relation
from viscurate.equivalence.taxonomy import classify
from viscurate.probes.build import load_probe
from viscurate.probes.manifest import ProbeManifest
from viscurate.skills.model import (
    ParamSpec,
    ParamsSchema,
    Skill,
    SkillMetadata,
)


# ==============================================================================================
# Clopper-Pearson
# ==============================================================================================
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI. Chosen over a normal approximation because the counts here are tiny and
    frequently 0 or n, where the normal interval collapses to zero width and lies."""
    from scipy.stats import beta

    if n == 0:
        return (0.0, 1.0)
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (low, high)


# ==============================================================================================
# Skill construction
# ==============================================================================================
def _make_fn(call: str, adapter: str, param_names: list[str]) -> Any:
    to_repr, from_repr = ADAPTERS[adapter]
    code = compile(call, f"<corpus_a:{call[:40]}>", "eval")

    def fn(image: np.ndarray, params: dict[str, Any], seed: int) -> np.ndarray:
        converted, _chain = to_repr(image)
        scope = dict(CALL_GLOBALS)
        scope["img"] = converted
        for name in param_names:
            scope[name] = params[name]
        return from_repr(eval(code, scope))

    return fn


def _spec_of(canonical: dict[str, Any]) -> tuple[ParamsSchema, list[str]]:
    specs = []
    for name, meta in (canonical or {}).items():
        grid = meta["grid"]
        specs.append(
            ParamSpec(
                name=name,
                type=meta["type"],
                default=grid[0],
                minimum=float(min(grid)),
                maximum=float(max(grid)),
            )
        )
    return ParamsSchema(params=tuple(specs)), [s.name for s in specs]


def build_skills(alignment: dict[str, Any]) -> tuple[dict[str, Skill], dict[str, dict[str, Any]]]:
    """One Skill per implementation (and per alternative variant). Returns (skills, metadata)."""
    skills: dict[str, Skill] = {}
    info: dict[str, dict[str, Any]] = {}
    for fam in alignment["families"]:
        family = fam["family"]
        schema, param_names = _spec_of(fam.get("canonical_params") or {})
        for impl in fam["implementations"]:
            entries = [(impl["id"], impl["call"], impl.get("alignment", "full"), None)]
            for var in impl.get("variants", []) or []:
                entries.append((var["id"], var["call"], "variant", impl["id"]))
            for sid, call, align, parent in entries:
                to_repr, _ = ADAPTERS[impl["adapter"]]
                skills[sid] = Skill(
                    id=sid,
                    # The comparator never sees name/description; they exist only so the A2 text
                    # baselines have the same surface a real library curator would read.
                    name=sid.split(".")[-1],
                    description=f"{family.replace('_', ' ')} via {sid}",
                    fn=_make_fn(call, impl["adapter"], param_names),
                    params_schema=schema,
                    metadata=SkillMetadata(family=family, trusted=True),
                )
                _, chain = to_repr(np.zeros((4, 4, 3), np.uint8))
                info[sid] = {
                    "family": family,
                    "library": sid.split(".")[0],
                    "adapter": impl["adapter"],
                    "conversion_chain": chain,
                    "alignment_status": align,
                    "variant_of": parent,
                    "call": call,
                    "notes": (impl.get("notes") or "").strip(),
                    "param_names": param_names,
                }
    return skills, info


def build_alignment_artifact(alignment: dict[str, Any], out: Path) -> ParamAlignment:
    """Emit a repo-format ParamAlignment: one axis per family over the canonical grid.

    Coupled canonical params (gaussian_blur's sigma AND ksize) are driven together through
    ``bindings`` keyed on the grid index, so the worst-case sweep evaluates both skills at the
    same *semantic* setting rather than at independently-varied params.
    """
    axes = []
    for fam in alignment["families"]:
        canonical = fam.get("canonical_params") or {}
        if not canonical:
            continue
        names = list(canonical)
        n = max(len(canonical[k]["grid"]) for k in names)
        members: dict[str, dict[str, Any]] = {}
        bindings: dict[str, dict[str, Any]] = {}
        for i in range(n):
            bindings[str(i)] = {
                k: canonical[k]["grid"][min(i, len(canonical[k]["grid"]) - 1)] for k in names
            }
        ids = [impl["id"] for impl in fam["implementations"]]
        ids += [v["id"] for impl in fam["implementations"] for v in impl.get("variants", []) or []]
        for sid in ids:
            members[sid] = {"bindings": bindings}
        axes.append({"name": f"{fam['family']}_grid", "values": list(range(n)), "members": members})
    doc = {"version": "1", "axes": axes, "subsumption_grids": {}}
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return ParamAlignment.model_validate(doc)


# ==============================================================================================
# Main
# ==============================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alignment", default="scripts/wacv_r2/corpus_a_alignment.yaml")
    ap.add_argument("--published-map", default="scripts/wacv_r2/corpus_a_published_map.yaml")
    ap.add_argument("--published-rows", default="results/wacv_r2/corpusA/published_rows.json")
    ap.add_argument("--probes-dir", default="data/probe_images_full")
    ap.add_argument(
        "--thresholds",
        required=True,
        help="ALREADY-CALIBRATED threshold file carried over unchanged; never re-fit here",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--out", default="results/wacv_r2/corpusA")
    ap.add_argument("--freeze-only", action="store_true", help="write the frozen pair list, stop")
    ap.add_argument("--max-cache-gb", type=float, default=24.0)
    ap.add_argument("--progress-every", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    align_path, pmap_path = Path(args.alignment), Path(args.published_map)
    alignment = yaml.safe_load(align_path.read_text(encoding="utf-8"))
    pmap = yaml.safe_load(pmap_path.read_text(encoding="utf-8"))

    skills, info = build_skills(alignment)
    families: dict[str, list[str]] = {}
    for sid, meta in info.items():
        families.setdefault(str(meta["family"]), []).append(sid)

    # ---- step 3: freeze the pair list BEFORE any execution -----------------------------------
    pairs: list[dict[str, Any]] = []
    for family, ids in sorted(families.items()):
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                pairs.append(
                    {
                        "family": family,
                        "a": a,
                        "b": b,
                        "lib_a": info[a]["library"],
                        "lib_b": info[b]["library"],
                        "cross_library": info[a]["library"] != info[b]["library"],
                        # A pair inherits the WEAKER of the two alignment statuses: comparing a
                        # partially-aligned implementation is a partially-aligned comparison.
                        "alignment_status": (
                            "variant"
                            if "variant"
                            in (info[a]["alignment_status"], info[b]["alignment_status"])
                            else (
                                "partial"
                                if "partial"
                                in (info[a]["alignment_status"], info[b]["alignment_status"])
                                else "full"
                            )
                        ),
                    }
                )
    self_pairs = sorted(skills)
    frozen = {
        "artifact": "corpus_a_frozen_pair_list",
        "n_implementations": len(skills),
        "n_pairs": len(pairs),
        "n_self_pairs": len(self_pairs),
        "alignment_sha256": hashlib.sha256(align_path.read_bytes()).hexdigest(),
        "published_map_sha256": hashlib.sha256(pmap_path.read_bytes()).hexdigest(),
        "implementations": info,
        "pairs": pairs,
        "self_pairs": self_pairs,
    }
    canonical = json.dumps(
        {"pairs": pairs, "self_pairs": self_pairs}, sort_keys=True, separators=(",", ":")
    )
    frozen["frozen_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    (out_dir / "frozen_pair_list.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(
        f"frozen pair list: {len(pairs)} pairs + {len(self_pairs)} self-pairs "
        f"over {len(skills)} implementations"
    )
    print(f"  frozen_sha256: {frozen['frozen_sha256']}")
    if args.freeze_only:
        return 0

    # ---- thresholds: carried over, never re-fit ------------------------------------------------
    thr_doc = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8"))
    thr_fields = thr_doc.get("thresholds", thr_doc)
    thresholds = ThresholdConfig.model_validate(thr_fields)
    if not getattr(thresholds, "calibrated", False):
        raise SystemExit(
            f"{args.thresholds} is not marked calibrated; Corpus A must run at an already-"
            "calibrated operating point (A1/01 §5 step 4). Refusing to run."
        )
    print(f"thresholds carried over from {args.thresholds} (calibrated=True)")

    manifest = ProbeManifest.model_validate_json(
        (Path(args.probes_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    battery = [(e.probe_id, load_probe(args.probes_dir, e.probe_id)) for e in manifest.entries]
    print(f"battery: {len(battery)} probes")
    param_alignment = build_alignment_artifact(alignment, out_dir / "corpus_a_param_alignment.yaml")

    from viscurate.equivalence.backends import ClipBackend, DinoBackend, LpipsBackend

    perceptual = LpipsBackend(device=args.device)
    semantic = DinoBackend(device=args.device)
    clip = ClipBackend(device=args.device) if args.clip else None

    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    t_start = time.time()

    for family, ids in sorted(families.items()):
        fam_skills = [skills[s] for s in sorted(ids)]
        provider = BatteryEvaluator(
            fam_skills,
            battery,
            seed=0,
            max_cache_entries=None,
            max_cache_bytes=int(args.max_cache_gb * 1_000_000_000),
        )
        # -- step 4: self-pair gate, per family ------------------------------------------------
        for sid in sorted(ids):
            view = skills[sid].comparator_view()
            try:
                res = classify(
                    view,
                    view,
                    provider,
                    thresholds=thresholds,
                    perceptual=perceptual,
                    semantic=semantic,
                    clip=clip,
                    alignment=param_alignment,
                    seed=0,
                )
                n_err = len(provider.outputs(sid).errors)
                gate_rows.append(
                    {
                        "impl": sid,
                        "family": family,
                        "relation": str(res.relation),
                        "passed": res.relation is Relation.EXACT,
                        "n_probe_errors": n_err,
                        "reason": res.reason,
                    }
                )
            except Exception as exc:
                gate_rows.append(
                    {
                        "impl": sid,
                        "family": family,
                        "relation": "ERROR",
                        "passed": False,
                        "n_probe_errors": -1,
                        "reason": f"{type(exc).__name__}: {exc}"[:300],
                    }
                )
                exclusions.append(
                    {
                        "unit": sid,
                        "stage": "self_pair_gate",
                        "reason": f"{type(exc).__name__}: {exc}"[:300],
                    }
                )

        # -- step 5: the pairs ------------------------------------------------------------------
        for pair in [p for p in pairs if p["family"] == family]:
            a, b = pair["a"], pair["b"]
            try:
                res = classify(
                    skills[a].comparator_view(),
                    skills[b].comparator_view(),
                    provider,
                    thresholds=thresholds,
                    perceptual=perceptual,
                    semantic=semantic,
                    clip=clip,
                    alignment=param_alignment,
                    seed=0,
                )
                rows.append(
                    {
                        **pair,
                        "relation": str(res.relation),
                        "direction": str(res.direction),
                        "licenses_merge": res.licenses_merge,
                        "reason": res.reason,
                        "worst_probe": res.worst_probe,
                        **{f"d_{k}": v for k, v in res.distances.items()},
                    }
                )
            except Exception as exc:
                rows.append(
                    {**pair, "relation": "ERROR", "reason": f"{type(exc).__name__}: {exc}"[:300]}
                )
                exclusions.append(
                    {
                        "unit": f"{a}|{b}",
                        "stage": "classify",
                        "reason": f"{type(exc).__name__}: {exc}"[:300],
                    }
                )
            if len(rows) % args.progress_every == 0:
                print(f"  [{len(rows)}/{len(pairs)}] {family} ({time.time() - t_start:.0f}s)")
        del provider

    for backend in (perceptual, semantic, clip):
        if backend is not None and hasattr(backend, "close"):
            backend.close()

    # ---- write per-pair records ---------------------------------------------------------------
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (out_dir / "pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "self_pair_gate.json").write_text(json.dumps(gate_rows, indent=2), encoding="utf-8")
    (out_dir / "exclusions.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")

    # ---- step 6: claim-vs-execution matrix over the joined published rows ----------------------
    published = json.loads(Path(args.published_rows).read_text(encoding="utf-8"))["rows"]
    pub_label = {
        (("kornia" if "kornia" in r["source_file"] else "torchvision"), r["left_name"]): r[
            "published_label"
        ]
        for r in published
    }
    by_pair = {(r["a"], r["b"]): r for r in rows}
    by_pair.update({(r["b"], r["a"]): r for r in rows})
    joined: list[dict[str, Any]] = []
    for jr in pmap["rows"]:
        key = (jr["page"], jr["left_name"])
        rec = by_pair.get((jr["left_impl"], jr["right_impl"]))
        if key not in pub_label or rec is None:
            exclusions.append(
                {"unit": f"{jr['page']}:{jr['left_name']}", "stage": "join", "reason": "unjoinable"}
            )
            continue
        joined.append(
            {
                "page": jr["page"],
                "published_class": jr["left_name"],
                "published_label": pub_label[key],
                "left_impl": jr["left_impl"],
                "right_impl": jr["right_impl"],
                "family": rec["family"],
                "alignment_status": rec["alignment_status"],
                "relation": rec["relation"],
                "licenses_merge": rec.get("licenses_merge", False),
                "reason": rec.get("reason", ""),
                "worst_probe": rec.get("worst_probe", ""),
            }
        )
    (out_dir / "claim_vs_execution.json").write_text(json.dumps(joined, indent=2), encoding="utf-8")

    # ``Relation`` is a StrEnum, so ``str(relation)`` is the bare value ("EXACT"), not
    # "Relation.EXACT" — keep these keys identical to what the rows actually carry.
    rel_order = [
        Relation.EXACT.value,
        Relation.PERCEPTUAL.value,
        Relation.SUBSUMPTION.value,
        Relation.SEMANTIC_PRESERVING.value,
        Relation.COMPLEMENTARY.value,
        Relation.UNCERTAIN.value,
        Relation.DISTINCT.value,
        "ERROR",
    ]
    matrix: dict[str, dict[str, int]] = {}
    for j in joined:
        matrix.setdefault(j["published_label"], dict.fromkeys(rel_order, 0))
        matrix[j["published_label"]][j["relation"]] = (
            matrix[j["published_label"]].get(j["relation"], 0) + 1
        )

    lines = ["# A1 Corpus A — cross-library functional-core audit", ""]
    lines.append(f"- implementations: **{len(skills)}** across **{len(families)}** families")
    lines.append(f"- within-family pairs scored: **{len(rows)}**")
    lines.append(f"- battery: **{len(battery)}** probes (frozen, reused from the synthetic study)")
    lines.append(f"- thresholds: carried over from `{args.thresholds}`, **not re-calibrated**")
    lines.append("")
    n_gate = len(gate_rows)
    n_gate_ok = sum(1 for g in gate_rows if g["passed"])
    lo, hi = clopper_pearson(n_gate_ok, n_gate)
    lines += [
        "## Self-pair sanity gate (harness validity)",
        "",
        f"Every implementation vs. itself must certify EXACT. **{n_gate_ok}/{n_gate}** pass "
        f"(exact 95% CI [{lo:.3f}, {hi:.3f}]).",
        "",
    ]
    if n_gate_ok < n_gate:
        lines += ["Failures — divergences involving these are NOT interpretable:", ""]
        lines += ["| impl | relation | probe errors | reason |", "|---|---|---:|---|"]
        for g in gate_rows:
            if not g["passed"]:
                lines.append(
                    f"| `{g['impl']}` | {g['relation']} | {g['n_probe_errors']} "
                    f"| {g['reason'][:90]} |"
                )
        lines.append("")

    lines += ["## Claim-vs-execution matrix (published rows only)", ""]
    header = "| published label | " + " | ".join(r.split(".")[-1] for r in rel_order) + " | n |"
    lines += [header, "|---" * (len(rel_order) + 2) + "|"]
    for label in ("direct", "partial", "none"):
        if label not in matrix:
            continue
        counts = matrix[label]
        n = sum(counts.values())
        lines.append(
            f"| {label} | " + " | ".join(str(counts.get(r, 0)) for r in rel_order) + f" | {n} |"
        )
    lines.append("")
    direct = [j for j in joined if j["published_label"] == "direct"]
    if direct:
        k = sum(1 for j in direct if j["licenses_merge"])
        lo, hi = clopper_pearson(k, len(direct))
        lines += [
            f"**Category-judgment survival rate (direct rows).** {k}/{len(direct)} = "
            f"{k / len(direct):.3f}, exact 95% CI [{lo:.3f}, {hi:.3f}] — the fraction of pairs a "
            "published table groups as the same operation that execution certifies as mergeable "
            "(EXACT ∪ PERCEPTUAL) at aligned parameters. Read as *a category-level judgment that "
            "does not survive execution*, never as a maintainer error.",
            "",
        ]
        full = [j for j in direct if j["alignment_status"] == "full"]
        if full:
            kf = sum(1 for j in full if j["licenses_merge"])
            lof, hif = clopper_pearson(kf, len(full))
            lines += [
                f"Restricted to FULL alignments only (partial alignments cannot separate a "
                f"behavioral difference from an alignment artifact): {kf}/{len(full)} = "
                f"{kf / len(full):.3f}, exact 95% CI [{lof:.3f}, {hif:.3f}].",
                "",
            ]
        lines += [
            "### Every direct row, enumerated",
            "",
            "| page | class | family | alignment | relation | deciding reason |",
            "|---|---|---|---|---|---|",
        ]
        for j in sorted(direct, key=lambda x: (x["relation"], x["family"])):
            lines.append(
                f"| {j['page']} | `{j['published_class']}` | {j['family']} | "
                f"{j['alignment_status']} | **{j['relation'].split('.')[-1]}** "
                f"| {j['reason'][:110]} |"
            )
        lines.append("")

    variant_rows = [r for r in rows if r["alignment_status"] == "variant"]
    lines += [
        "## Alignment sensitivity (alternative defensible alignments)",
        "",
        f"{len(variant_rows)} pairs involve an alternative alignment variant. A divergence visible "
        "under only one variant is an alignment artifact, not a behavioral finding.",
        "",
    ]
    if variant_rows:
        lines += ["| family | a | b | relation |", "|---|---|---|---|"]
        for r in sorted(variant_rows, key=lambda x: x["family"]):
            lines.append(
                f"| {r['family']} | `{r['a']}` | `{r['b']}` | {r['relation'].split('.')[-1]} |"
            )
        lines.append("")

    rel_counts: dict[str, int] = {}
    for r in rows:
        rel_counts[r["relation"]] = rel_counts.get(r["relation"], 0) + 1
    lines += ["## All within-family pairs, by relation", "", "| relation | n |", "|---|---:|"]
    for rel in rel_order:
        if rel in rel_counts:
            lines.append(f"| {rel.split('.')[-1]} | {rel_counts[rel]} |")
    lines.append("")
    lines += [
        f"## Exclusions ({len(exclusions)})",
        "",
        "| unit | stage | reason |",
        "|---|---|---|",
    ]
    for e in exclusions[:60]:
        lines.append(f"| `{e['unit']}` | {e['stage']} | {e['reason'][:100]} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "artifact": "corpus_a_run",
        "frozen_pair_list_sha256": frozen["frozen_sha256"],
        "alignment_sha256": frozen["alignment_sha256"],
        "published_map_sha256": frozen["published_map_sha256"],
        "thresholds_file": args.thresholds,
        "thresholds": thr_fields,
        "recalibrated_on_corpus_a": False,
        "battery_dir": args.probes_dir,
        "battery_n": len(battery),
        "device": args.device,
        "clip": bool(args.clip),
        "n_implementations": len(skills),
        "n_pairs_scored": len(rows),
        "self_pair_gate_pass": n_gate_ok,
        "self_pair_gate_total": n_gate,
        "n_exclusions": len(exclusions),
        "library_versions": _library_versions(),
        "albucore_shim": ALBUCORE_SHIM,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "exclusions.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    print(f"corpus A: {len(rows)} pairs, gate {n_gate_ok}/{n_gate} -> {out_dir}/report.md")
    return 0


def _library_versions() -> dict[str, str]:
    import albumentations
    import cv2
    import kornia
    import PIL
    import skimage
    import torch
    import torchvision

    return {
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "scikit-image": skimage.__version__,
        "kornia": kornia.__version__,
        "albumentations": albumentations.__version__,
        "torchvision": torchvision.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


if __name__ == "__main__":
    raise SystemExit(main())

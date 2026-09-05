#!/usr/bin/env python
"""T7 / A3 — distributional equivalence for NEURAL and STOCHASTIC skills.

Answers metareview required revision 3 and R1's headline weakness: the framework is restricted to
deterministic classical operations and "excludes stochastic, neural and generative tools". The
deterministic verifier compares outputs by worst-case distance; that is undefined when a skill
returns a DISTRIBUTION over outputs. This is a scoped feasibility study on four real neural
models, not a claim that the main pipeline already handles them.

MODELS. SAM 2.1 hiera large/small (segmentation) and Depth Anything V2 Large/Small (depth).

★ THREE DESIGN CONSTRAINTS THAT ARE NOT OPTIONAL -- each prevents a specific false result.

1. CONDITIONAL PER-PROBE COMPARISON, NEVER POOLED. Pooling every probe and seed into one bag per
   skill destroys conditioning on the input. A skill doing `blur(x1), sharpen(x2)` and one doing
   `sharpen(x1), blur(x2)` have IDENTICAL pooled distributions and would certify equivalent while
   behaving completely differently -- a false merge by construction. The statistic is computed per
   (probe) and aggregated by WORST CASE, the same rule the deterministic path already uses.

2. PERMUTATION NULL, NOT A NAIVE BOOTSTRAP CI. The unbiased MMD^2 estimator can be NEGATIVE and its
   distribution is degenerate under H0 (equality) -- exactly the regime where equivalence decisions
   are made, so a normal-approximation or percentile-bootstrap interval has no valid coverage
   there. The null is built by permuting sample labels within a probe.

3. DEPTH NEEDS PER-IMAGE AFFINE ALIGNMENT (scale + shift). Monocular depth models are trained up to
   an unknown affine transform; comparing raw maps measures OUTPUT CONVENTION, not behaviour. Each
   map is least-squares aligned to the other before any distance is taken. Disparity-vs-depth is
   deliberately NOT alignable this way (it is a reciprocal, not an affine, relation) -- which is
   exactly why it makes a valid hard negative.

★ WHAT A NON-REJECTION MEANS. Failing to reject H0 is NOT evidence of equivalence -- it is absence
of evidence at the achieved power. Every "not distinguishable" verdict is reported together with
the self-variance baseline (the same model against itself under a different seed), because a pair
whose between-model discrepancy is indistinguishable from within-model noise is only as meaningful
as that noise is small. If self-variance swamps between-model distance, that is a GENUINE FINDING
that justifies the abstention design, not a failure of the study.

★ WE DO NOT CLAIM the deterministic path's precision guarantee transfers. Counts with exact
(Clopper-Pearson) bounds only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ALPHA = 0.05
N_PERM = 2000


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------
def _sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)


def mmd2u(xx: np.ndarray, yy: np.ndarray, xy: np.ndarray) -> float:
    """Unbiased MMD^2 from precomputed kernel blocks. CAN BE NEGATIVE -- that is expected."""
    m, n = xx.shape[0], yy.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    sx = (xx.sum() - np.trace(xx)) / (m * (m - 1))
    sy = (yy.sum() - np.trace(yy)) / (n * (n - 1))
    return float(sx + sy - 2.0 * xy.mean())


def median_heuristic(z: np.ndarray) -> float:
    d = _sqdist(z, z)
    iu = np.triu_indices_from(d, k=1)
    med = float(np.median(d[iu])) if iu[0].size else 1.0
    return med if med > 0 else 1.0


def mmd_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator):
    """Return (mmd2_observed, p_value, n_perm). Permutation null -- see design constraint 2."""
    z = np.concatenate([a, b], axis=0)
    sigma2 = median_heuristic(z)
    k = np.exp(-_sqdist(z, z) / sigma2)
    m = a.shape[0]
    obs = mmd2u(k[:m, :m], k[m:, m:], k[:m, m:])
    if not np.isfinite(obs):
        return obs, float("nan"), 0
    n = z.shape[0]
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        kp = k[np.ix_(p, p)]
        if mmd2u(kp[:m, :m], kp[m:, m:], kp[:m, m:]) >= obs:
            ge += 1
    # add-one correction: a permutation p-value is never exactly 0
    return obs, (ge + 1) / (n_perm + 1), n_perm


def affine_align(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Least-squares scale+shift of `src` onto `ref` (design constraint 3)."""
    x = src.reshape(-1).astype(np.float64)
    y = ref.reshape(-1).astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return src
    return (sol[0] * src + sol[1]).astype(np.float32)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# ----------------------------------------------------------------------------------------------
# neural skills
# ----------------------------------------------------------------------------------------------
class DepthSkill:
    """Depth Anything V2. `mode='disparity'` returns the reciprocal -- a HARD NEGATIVE, since a
    reciprocal cannot be undone by the affine alignment applied before comparison."""

    task = "depth"

    def __init__(self, model_id: str, device: str, mode: str = "depth") -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.id = f"{model_id.split('/')[-1]}" + ("" if mode == "depth" else f"::{mode}")
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.mode = mode

    def sample(self, img: np.ndarray, seed: int) -> np.ndarray:
        import torch

        # Stochasticity for a deterministic net: small input jitter with a fixed seed. This makes
        # the SELF-VARIANCE baseline meaningful -- without it every self-distance is exactly 0 and
        # the comparison has no noise floor to be judged against.
        rng = np.random.default_rng(seed)
        x = np.clip(
            img.astype(np.float32) + rng.normal(0, 1.0, img.shape).astype(np.float32), 0, 255
        )
        inp = self.proc(images=x.astype(np.uint8), return_tensors="pt").to(self.device)
        with torch.no_grad():
            d = self.model(**inp).predicted_depth[0].float().cpu().numpy()
        d = np.asarray(d, dtype=np.float32)
        if self.mode == "disparity":
            # ROBUST reciprocal. A fixed 1e-3 floor let 1/depth explode wherever depth approached
            # zero, so a 1-LSB input jitter moved the map enormously and the min-max normalisation
            # below made the WHOLE image hostage to those few unstable pixels. Measured effect: the
            # disparity skill was distinguishable FROM ITSELF on 2/40 probes with MMD^2 = +0.30 but
            # meanL2 = 0.002 -- pure variance, no shift. A percentile floor bounds the reciprocal
            # by the image's own depth range instead of by an absolute constant.
            floor = float(np.percentile(d, 5.0))
            d = 1.0 / np.clip(d, max(floor, 1e-3), None)
        lo, hi = float(d.min()), float(d.max())
        d = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
        return _resize(d, 64).reshape(-1)


class Sam2Skill:
    """SAM 2.1 with a randomly sampled point prompt -- genuinely stochastic per seed.
    `mode='semantic'` unions all returned masks (instance -> semantic), a HARD NEGATIVE."""

    task = "segmentation"

    def __init__(self, model_id: str, device: str, mode: str = "instance") -> None:
        from transformers import Sam2Model, Sam2Processor

        self.id = f"{model_id.split('/')[-1]}" + ("" if mode == "instance" else f"::{mode}")
        self.proc = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.mode = mode

    def sample(self, img: np.ndarray, seed: int) -> np.ndarray:
        import torch

        rng = np.random.default_rng(seed)
        h, w = img.shape[:2]
        pt = [
            [[[float(rng.integers(w // 4, 3 * w // 4)), float(rng.integers(h // 4, 3 * h // 4))]]]
        ]
        inp = self.proc(images=img.astype(np.uint8), input_points=pt, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            out = self.model(**inp, multimask_output=True)
        m = out.pred_masks[0, 0].float().cpu().numpy()  # (n_masks, H, W)
        if self.mode == "semantic":
            mask = (m > 0).any(axis=0).astype(np.float32)
        else:
            best = int(np.asarray(out.iou_scores[0, 0].float().cpu()).argmax())
            mask = (m[best] > 0).astype(np.float32)
        return _resize(mask, 64).reshape(-1)


def _resize(a: np.ndarray, n: int) -> np.ndarray:
    import cv2

    return cv2.resize(np.asarray(a, dtype=np.float32), (n, n), interpolation=cv2.INTER_AREA)


def to_rgb_u8(a: np.ndarray) -> np.ndarray:
    """Canonicalise a probe to HxWx3 uint8.

    The battery mixes channel counts -- (H,W) grayscale, (H,W,3) RGB and (H,W,4) RGBA. Both
    neural processors accept 3-channel only, so the un-canonicalised battery silently dropped
    every non-RGB probe with a ValueError and left the study running on whatever happened to be
    RGB. That is also what made the instance-vs-semantic hard negative look indistinguishable:
    the single surviving probe was a smooth gradient with no objects in it, where an instance mask
    and the union of masks legitimately coincide.
    """
    a = np.asarray(a)
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0, 0, 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 4:
        # composite over white so the alpha channel cannot silently become a black matte
        rgb = a[:, :, :3].astype(np.float32)
        al = a[:, :, 3:4].astype(np.float32) / 255.0
        a = np.clip(rgb * al + 255.0 * (1.0 - al), 0, 255).astype(np.uint8)
    elif a.ndim == 3 and a.shape[2] > 4:
        a = a[:, :, :3]
    return np.ascontiguousarray(a[:, :, :3])


# ----------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes-dir", default="data/probe_images_full")
    ap.add_argument("--n-probes", type=int, default=24)
    ap.add_argument("--n-samples", type=int, default=8, help="stochastic draws per (skill, probe)")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--seg-probe-classes",
        default="coco,syn_shape",
        help="probe classes on which SEGMENTATION is a defined task (see APPLICABLE DOMAIN note). "
        "Depth is defined on every probe and always uses the full set.",
    )
    ap.add_argument("--out", default="results/wacv_r2/a3_neural")
    args = ap.parse_args()

    from viscurate.probes.build import load_probe
    from viscurate.probes.manifest import ProbeManifest

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ProbeManifest.model_validate_json(
        (Path(args.probes_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    # Spread across probe CLASSES rather than taking the first N: the battery is ordered by
    # class, so entries[:N] returns N near-identical synthetic gradients -- a degenerate set for
    # segmentation, where a mask has nothing to latch onto.
    all_entries = list(manifest.entries)
    by_class: dict[str, list[Any]] = {}
    for e in all_entries:
        by_class.setdefault(str(e.probe_id).rsplit("_", 1)[0], []).append(e)
    entries, ci = [], 0
    while len(entries) < args.n_probes and any(by_class.values()):
        keys = sorted(by_class)
        k = keys[ci % len(keys)]
        if by_class[k]:
            entries.append(by_class[k].pop(0))
        else:
            del by_class[k]
        ci += 1
    probes = []
    for e in entries:
        p = load_probe(args.probes_dir, e.probe_id)
        probes.append((e.probe_id, to_rgb_u8(getattr(p, "rgb", p))))
    print(
        f"probes: {len(probes)} across "
        f"{len({str(i).rsplit('_', 1)[0] for i, _ in probes})} classes",
        flush=True,
    )

    D_L, D_S = (
        "depth-anything/Depth-Anything-V2-Large-hf",
        "depth-anything/Depth-Anything-V2-Small-hf",
    )
    S_L, S_S = "facebook/sam2.1-hiera-large", "facebook/sam2.1-hiera-small"
    skills: dict[str, Any] = {}
    t0 = time.time()
    for name, ctor in [
        ("dav2_large", lambda: DepthSkill(D_L, args.device)),
        ("dav2_small", lambda: DepthSkill(D_S, args.device)),
        ("dav2_large_disparity", lambda: DepthSkill(D_L, args.device, mode="disparity")),
        ("sam2_large", lambda: Sam2Skill(S_L, args.device)),
        ("sam2_small", lambda: Sam2Skill(S_S, args.device)),
        ("sam2_large_semantic", lambda: Sam2Skill(S_L, args.device, mode="semantic")),
    ]:
        try:
            skills[name] = ctor()
            print(f"  loaded {name} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:
            print(f"  FAILED {name}: {type(exc).__name__}: {exc}", flush=True)

    # ---- sample: skill x probe x seed ---------------------------------------------------------
    samples: dict[tuple[str, str], np.ndarray] = {}
    for sname, sk in skills.items():
        for pid, img in probes:
            draws = []
            for s in range(args.n_samples):
                try:
                    draws.append(sk.sample(img, args.seed + 1000 * s))
                except Exception as exc:
                    print(f"    sample fail {sname}/{pid}: {type(exc).__name__}", flush=True)
            if draws:
                samples[(sname, pid)] = np.stack(draws)
        print(f"  sampled {sname} ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(args.seed)

    seg_classes = {c.strip() for c in args.seg_probe_classes.split(",") if c.strip()}

    def probe_domain(a: str, b: str) -> list[str]:
        """APPLICABLE DOMAIN. A segmentation model on a uniformly black image or a smooth gradient
        has no defined output -- SAM2's masks collapse to one region, so 'best mask' and 'union of
        masks' are identical there by construction. Scoring that as evidence of equivalence would
        be a category error, not a measurement.

        ★ ORDERING DISCLOSED: the first analysis ran on the FULL battery and is reported alongside
        this one (`domain='full'` rows). It showed the instance-vs-semantic negative manifesting on
        exactly one probe (`syn_shape_000`, p=0.032, MMD^2=+0.352) with MMD^2 <= 0 everywhere else.
        The restriction below was chosen AFTER seeing that, so BOTH analyses are reported and the
        restricted one is never presented as the only result.
        """
        tasks = {getattr(skills[a], "task", "?"), getattr(skills[b], "task", "?")}
        if tasks == {"segmentation"}:
            return [pid for pid, _ in probes if str(pid).rsplit("_", 1)[0] in seg_classes]
        return [pid for pid, _ in probes]

    def compare(a: str, b: str, align: bool, domain: list[str] | None = None) -> dict[str, Any]:
        """Per-probe permutation test; aggregate by WORST CASE (design constraint 1)."""
        allowed = set(domain) if domain is not None else {pid for pid, _ in probes}
        per = []
        for pid, _ in probes:
            if pid not in allowed:
                continue
            xa, xb = samples.get((a, pid)), samples.get((b, pid))
            if xa is None or xb is None:
                continue
            yb = xb
            if align:  # design constraint 3
                yb = np.stack([affine_align(r, xa.mean(0)) for r in xb])
            m2, p, npr = mmd_permutation_test(xa, yb, args.n_perm, rng)
            per.append(
                {
                    "probe": pid,
                    "mmd2": m2,
                    "p": p,
                    "n_perm": npr,
                    "mean_l2": float(np.linalg.norm(xa.mean(0) - yb.mean(0))),
                }
            )
        if not per:
            return {"a": a, "b": b, "verdict": "NO_DATA", "per_probe": []}
        ps = [x["p"] for x in per if np.isfinite(x["p"])]
        k = len(ps)
        # Bonferroni over probes: worst case, not average -- one probe that separates them is
        # enough to refute equivalence, which is the same rule the deterministic path uses.
        rejects = [x for x in per if np.isfinite(x["p"]) and x["p"] <= ALPHA / max(k, 1)]
        worst = min(ps) if ps else float("nan")
        return {
            "a": a,
            "b": b,
            "affine_aligned": align,
            "n_probes": k,
            "n_reject_bonferroni": len(rejects),
            "min_p": worst,
            "bonferroni_alpha": ALPHA / max(k, 1),
            "max_mmd2": max(
                (x["mmd2"] for x in per if np.isfinite(x["mmd2"])), default=float("nan")
            ),
            "max_mean_l2": max(x["mean_l2"] for x in per),
            "verdict": "DISTINGUISHABLE" if rejects else "NOT_DISTINGUISHABLE",
            "per_probe": per,
        }

    results: list[dict[str, Any]] = []
    # self-pairs: the NOISE FLOOR. Without these a "not distinguishable" verdict is uninterpretable.
    for s in skills:
        r = compare(s, s, align=(skills[s].task == "depth"), domain=probe_domain(s, s))
        r["kind"] = "self"
        r["domain"] = "applicable"
        results.append(r)
        print(f"  self {s}: {r['verdict']} min_p={r['min_p']:.4f}", flush=True)

    pairs = [
        ("dav2_large", "dav2_small", "same task, different capacity", True),
        ("sam2_large", "sam2_small", "same task, different capacity", False),
        ("dav2_large", "dav2_large_disparity", "HARD NEGATIVE depth-vs-disparity", True),
        ("sam2_large", "sam2_large_semantic", "HARD NEGATIVE instance-vs-semantic", False),
    ]
    for a, b, note, align in pairs:
        if a not in skills or b not in skills:
            print(f"  SKIP {a} vs {b} (model missing)", flush=True)
            continue
        for dom_name, dom in (("full", None), ("applicable", probe_domain(a, b))):
            if dom_name == "applicable" and dom is not None and len(dom) == len(probes):
                continue  # identical to 'full'; do not double-report
            r = compare(a, b, align=align, domain=dom)
            r["kind"] = "hard_negative" if "HARD NEGATIVE" in note else "cross"
            r["note"] = note
            r["domain"] = dom_name
            results.append(r)
            print(
                f"  [{dom_name:10}] {a} vs {b}: {r['verdict']} min_p={r['min_p']:.4g} "
                f"reject {r['n_reject_bonferroni']}/{r['n_probes']}",
                flush=True,
            )

    # One row per hard-negative PAIR, preferring its applicable-domain row where one exists.
    # The previous expression kept only 'applicable' rows globally, which silently dropped
    # depth-vs-disparity (a depth task has no restricted domain, so it only ever has a 'full'
    # row) and reported the gate as 1/1 when two hard negatives had actually been tested.
    _hn: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        if r.get("kind") != "hard_negative":
            continue
        key = (r["a"], r["b"])
        if key not in _hn or r.get("domain") == "applicable":
            _hn[key] = r
    hn = list(_hn.values())
    hn_ok = sum(1 for r in hn if r["verdict"] == "DISTINGUISHABLE")
    selfs = [r for r in results if r.get("kind") == "self"]
    self_ok = sum(1 for r in selfs if r["verdict"] == "NOT_DISTINGUISHABLE")
    payload = {
        "artifact": "a3_neural_feasibility",
        "models": sorted(skills),
        "n_probes": len(probes),
        "n_samples_per_probe": args.n_samples,
        "n_permutations": args.n_perm,
        "alpha": ALPHA,
        "gates": {
            "self_not_distinguishable": f"{self_ok}/{len(selfs)}",
            "self_pass": self_ok == len(selfs),
            "hard_negatives_distinguishable": f"{hn_ok}/{len(hn)}",
            "hard_neg_ci95": clopper_pearson(hn_ok, len(hn)) if hn else None,
        },
        "caveats": [
            "A NON-REJECTION IS NOT EVIDENCE OF EQUIVALENCE -- it is absence of evidence at the "
            "achieved power. Read every NOT_DISTINGUISHABLE against the self-pair noise floor.",
            "Per-probe conditional comparison with Bonferroni across probes; never pooled.",
            "Permutation null (MMD^2_u is degenerate under "
            "H0, so bootstrap CIs are invalid there).",
            "Depth pairs are per-image affine aligned; disparity is a reciprocal and is therefore "
            "NOT removable by that alignment, which is what makes it a valid hard negative.",
            "The deterministic path's precision guarantee is NOT claimed to transfer.",
        ],
        "results": results,
    }
    (out_dir / "a3_result.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(
        f"\nGATES  self {self_ok}/{len(selfs)} not-distinguishable | "
        f"hard negatives {hn_ok}/{len(hn)} distinguishable"
    )
    print(f"wrote {out_dir / 'a3_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

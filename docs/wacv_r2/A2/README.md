# A2 — Text/Embedding Baselines + Verifier Self-Validation

**AC required revision #2:** *"Add a text/embedding baseline and verifier self-validation"*
**Also answers:** R3 W2 (central claim never tested) · R3 W3 (verifier not validated) · R3 minor (no probe analysis) · R2 (OOD, security, external validity)

---

## ★ BLOCKER first — the merge-policy contradiction

`SEMANTIC-PRESERVING` is described **four inconsistent ways** across the manuscript, and one of them is the premise justifying the binary `mergeable` axis on which every method in Table 2 is compared. **The positive class every safety metric is computed against is therefore unsettled.**

Read the gate implementation before running anything in this folder. Full detail: [`02_verifier_self_validation.md`](02_verifier_self_validation.md) §0 and [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §1.

---

## ★ Status update 2026-08-19 — the "partly done" claim below is now disputed, not confirmed

> See [`../CODEBASE_RECONCILIATION.md`](../CODEBASE_RECONCILIATION.md) §7 for the full evidence. Short version: `phase4.md` in the newer repo (https://github.com/farhanishraq17/VisCurate-) presents Table 2/3's exact numbers — `0/926`, `0/6`, `SEMANTIC_PRESERVING` precision `0.037`, etc. — as a completed real-backend run. `WORK_PROCESS_OVERVIEW.md`, **committed in the same push**, states explicitly that the only confirmed completed run had **null LPIPS/DINO/CLIP backends, 24 pairs (not 944), uncalibrated thresholds, on CPU**, and lists "the visual/text divergence result was finalized" under things the paper must **not** claim.
>
> **These two documents cannot both be right.** Until someone checks `results/` on the machine that produced `phase4.md` for a manifest showing real backends + `calibrated: true`, treat every ✅ below as *"code exists and a run was attempted,"* not *"this number is confirmed real."* The row-level checkmarks are being kept because the **code** described is genuinely built — only the **result rows** (Table 2, Table 3) are in dispute.

## Status: machinery is built; know what's built before you start

The Aug-15 draft is **not** the reviewed version. Someone already added `sec/6_methodology.tex` §4.4 "Equivalence-Verification Benchmark" with three text baselines and two result tables. Do not redo the *code*. Do not assume the *result numbers* are settled — see the box above.

| Component | State |
|---|---|
| `name-match` baseline (Jaccard, τ=0.5) | ✅ Built |
| `embedding-cosine` baseline (TF–IDF, τ=0.6) | ✅ Built |
| `llm-on-descriptions` baseline | ✅ Built |
| Safety/divergence table (Table 2) | ⚠️ Code built; **the specific `0/926` / `0/6` numbers are disputed — see box above** |
| Per-relation verifier quality (Table 3) | ⚠️ Code built; **same dispute**, and separately, three rows are unusable regardless (see below) |
| Sentence-embedding baseline | ❌ Missing |
| **Code-embedding baseline** | ❌ Missing — and the paper explicitly claims source code is unreliable |
| **AST/clone-detection baseline** | ❌ Missing |
| **LLM-on-source-code baseline** | ❌ Missing |
| **Merge-relation recall** (`EXACT`/`PERCEPTUAL`) | ❌ **Not estimable on `G_0`** |
| Threshold sensitivity / backend ablation | ❌ Missing |
| Probe-battery ablation | ❌ Missing |
| OOD divergence | ❌ Missing |
| Side-effect channel | ❌ Missing |

---

## Files

| File | Contents |
|---|---|
| [`01_baseline_ladder.md`](01_baseline_ladder.md) | The eight-rung baseline ladder, matched-precision comparison protocol |
| [`02_verifier_self_validation.md`](02_verifier_self_validation.md) | The `G_ρ` re-analysis — **free, do first**; fixes the merge-recall hole |
| [`03_ablations.md`](03_ablations.md) | Backend, threshold, seed, and probe-battery ablations |
| [`04_ood_and_scope_limits.md`](04_ood_and_scope_limits.md) | OOD divergence + side-effect channel (R2's structural critiques) |

---

## The three holes in the current draft

### Hole 1 — no code-based baseline, but the paper claims code is unreliable
`sec/0_abstract.tex` and `sec/1_intro.tex:15` both assert that a skill's *"name, description, or source code"* is a poor proxy. Only name and description are tested. **The source-code half of the claim is asserted and never evaluated.** → [`01_baseline_ladder.md`](01_baseline_ladder.md)

### Hole 2 — merge recall is not estimable
`sec/7_results.tex:60` states plainly: *"`EXACT` and `PERCEPTUAL` have no support in `G_0`, since a clean library contains no true duplicates by construction, so the mergeable positive class is not estimable from this graph."*

So the verifier's precision and recall **on exactly the two relations that authorize a destructive merge** are unreported. R3 asked for verifier validation; this is the half that is missing, and R3 has the expertise to spot it.

**The fix is free.** The corrupted graphs `G_ρ` *inject* exact duplicates (defect type iii) and subsumptions (type iv) with known ground truth. The positive class exists there by construction. This is pure re-analysis of runs already completed. → [`02_verifier_self_validation.md`](02_verifier_self_validation.md) — **do this in week 0.**

### Hole 3 — two broken rows in Table 3
| Row | Value | Problem |
|---|---|---|
| `SEMANTIC-PRESERVING` precision | **0.037** | indefensible as a point estimate |
| `COMPLEMENTARY` F1 | **0.000** | support of one pair — the metric is meaningless |

A 0.000 in a results table invites a reviewer to assume the whole method is broken, regardless of the surrounding argument. → [`02_verifier_self_validation.md`](02_verifier_self_validation.md) §4

---

## Priority within A2

| Order | Task | Effort | Why |
|---|---|---|---|
| **1** | `G_ρ` re-analysis (Hole 2) | **Free** — no new compute | Fills the biggest gap at zero cost |
| **2** | Fix broken rows (Hole 3) | Low | Removes the two numbers most likely to sink the paper on sight |
| **3** | Code + AST + LLM-source baselines (Hole 1) | Medium | Closes the paper's own claim |
| **4** | Probe-battery ablation | Low | R3 minor; feeds A4 cost reduction |
| **5** | Backend/threshold/seed ablations | Medium | Robustness |
| **6** | OOD divergence | Medium | Converts R2's unanswerable objection into a measured number |
| **7** | Side-effect channel | Medium | Turns R2's structural criticism into a contribution |

---

## Checklist

- [ ] Re-analyze `G_ρ` for merge-relation P/R/F1 (**week 0, free**)
- [ ] Rebuild `SEMANTIC-PRESERVING` as a P–R curve, not a point
- [ ] Build dedicated complementary probe set (20–30 pairs) or remove the row
- [ ] Add sentence-embedding baseline
- [ ] Add code-embedding baseline
- [ ] Add AST clone-detection baseline
- [ ] Add LLM-on-source-code baseline
- [ ] Re-run all baselines at **matched precision**
- [ ] Probe-battery ablation → report the knee → hand the number to A4
- [ ] Backend ablation (LPIPS/DINO/CLIP/pixel/full)
- [ ] Threshold sensitivity sweeps
- [ ] Re-calibrate on 5 seeds; report mean ± std
- [ ] OOD battery + certificate-flip rate
- [ ] Trojan defect type + syscall channel
- [ ] Re-run baselines on A1's real corpora

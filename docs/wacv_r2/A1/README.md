# A1 — Real, Uncurated Open-Source Library Audit

**AC required revision #1:** *"Evaluation on at least one real, uncurated open-source skill library"*
**Also answers:** R1 W2 (reliance on synthetic corruption) · R2 W2 + sole revision request (external validity) · R3 W1 (entirely synthetic evaluation)

**Priority: highest in the suite.** The only weakness all three reviewers raised independently.

---

## Files

| File | Contents |
|---|---|
| [`01_corpus_A_cross_library.md`](01_corpus_A_cross_library.md) | Cross-library claimed-equivalence audit — **do this first** |
| [`02_corpus_B_comfyui.md`](02_corpus_B_comfyui.md) | ComfyUI custom-node ecosystem audit |
| [`03_annotation_and_metrics.md`](03_annotation_and_metrics.md) | Human annotation protocol, all metrics, all output tables/figures |

Background survey with sources: [`../../Viscurarte_Rebuttal/A1_REAL_LIBRARY_PLAN.md`](../../Viscurarte_Rebuttal/A1_REAL_LIBRARY_PLAN.md)

---

## What A1 must prove

Three separable claims requiring different corpora:

| Claim | Needs | Corpus |
|---|---|---|
| **C1.** Description-derived equivalence fails on real code, not just injected defects | Real, **externally authored** equivalence claims | **A** |
| **C2.** Real libraries decay the way our seven defect types model | A large, genuinely uncurated community library | **B** |
| **C3.** VisCurate runs on code we did not write | Both | A + B |

---

## ⚠ Corpus B is mandatory — revised 2026-08-18

The original plan treated Corpus B as "scoped-but-cuttable" with Corpus A as a sufficient fallback. **That was wrong** ([`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §R4).

OpenCV, Kornia, Albumentations, and torchvision are each **individually curated** — well-maintained libraries with review processes. Corpus A tests cross-library equivalence judgments, which is valuable and answers R3, but **it is not "an uncurated open-source skill library."** The AC's item #1 is worded specifically, and only Corpus B satisfies it literally.

| Corpus | Satisfies AC #1? | Answers R3's "central claim untested"? |
|---|---|---|
| A (cross-library) | **No** — these libraries are curated | **Yes** |
| B (ComfyUI) | **Yes** | Partly |

**Contingency is reduced scale, not omission.** Better to audit 25 packs with isolated environments, real annotation, and full provenance than 100 packs fragilely. A small, rigorous, genuinely-uncurated audit answers the AC; a large sloppy one does not.

---

## Corpora selected

### Corpus A — cross-library category-level judgments (LOW effort, HIGHEST signal)
Albumentations / Kornia / torchvision v2 / OpenCV / Pillow / scikit-image.

The Albumentations maintainers publish transform-by-transform equivalence tables: **~94 pairs to Torchvision v2**, **75 pairs to Kornia**, each labelled *direct equivalent* / *partial* / *no equivalent*.

**This is a description-based equivalence oracle authored by domain experts, published externally, with zero involvement from us.** It is immune to the "you built the baseline to lose" objection that any self-constructed baseline invites — which is precisely R3's complaint that the central claim "is never actually demonstrated experimentally."

### Corpus B — ComfyUI custom nodes (HIGH effort, high signal)
**5,192 node packs** in the Comfy Registry. Community-contributed, no central review, genuine decay. A ComfyUI node is nearly a literal implementation of the paper's skill tuple `⟨k, μ, d, f_s, Θ_s⟩`.

### Rejected
| Candidate | Why rejected |
|---|---|
| Agent skill marketplaces (SkillsMP, 2.6M `SKILL.md` files) | Markdown instruction files, not executable image ops. Usable for one motivating sentence only. |
| `anthropics/skills` (17 examples) | Too small, curated, wrong modality |
| VisProg / ViperGPT / Chameleon tool sets | Tens of tools, hand-authored by paper authors — too curated to carry C2 |

---

## Timeline

| Week | Corpus A | Corpus B |
|---|---|---|
| **W0** | Freeze pair list; hand-author parameter alignment maps; **hash and commit before any execution** | Freeze registry pack list w/ commit SHAs; venv strategy; **recruit annotators** |
| **W1** | Execute; produce T2, T3; enumerate over-claimed pairs; draft F1 | Headless extraction shim; first extraction pass; exclusion log |
| **W2** | Adjudicate disagreements (small n); finalize tables | Execute at scale; fingerprint + verify; T1, T5 |
| **W3** | Write §5.x Corpus A subsection | Launch stratified annotation batch |
| **W4** | — | Adjudicate, compute α, produce T4, F2; write Corpus B subsection |

**Corpus A is fully done by end of W2.** If everything else slips, Corpus A alone plus A2's free `G_ρ` re-analysis is a defensible response to AC items #1 and #2.

---

## Go/no-go checkpoints

| When | Check | If it fails |
|---|---|---|
| End W0 | Parameter alignment maps authored for ≥15 operator families | Cut Corpus A scope to the highest-yield families; do not slip the freeze |
| End W1 | ≥60% of Corpus A pairs execute successfully on both sides | Investigate — likely a canonicalization bug, not a corpus problem |
| End W1 | ≥30 ComfyUI packs import cleanly in the harness | Cut Corpus B to the cleanly-importing subset and report the install-failure rate as a finding |
| End W2 | Corpus A produces a non-empty over-claim set | Still publishable — see Risks below |
| End W2 | Corpus B yields ≥300 executable image→image skills | Drop to top-50 packs; report honestly |

---

## Risks

| Risk | Mitigation |
|---|---|
| ComfyUI dependency hell consumes the schedule | The primitive-inputs-only filter (`02_corpus_B_comfyui.md` §3) removes most of it. **Hard-cap Corpus B at 2 weeks — but reduce scale, do not drop it.** Corpus B is the only corpus satisfying AC #1. |
| Corpus A over-claim rate is low (maintainers mostly right) | Still answers AC #1, and *strengthens* credibility: "execution confirms expert judgment in X% of cases and localizes the Y% it misses, which no text method can." **Do not tune to inflate the number.** |
| VisCurate performs worse on real than synthetic | Report it. The reviewers already suspect a generalization gap. **Never re-calibrate on real-corpus data.** |
| Parameter alignment contested ("you aligned them wrong") | Publish the maps; author from docs only; pre-register and hash; report divergence across the whole grid, not one point |
| Registry contents shift mid-experiment | Pin commit SHAs at W0. Non-negotiable. |
| Annotator recruitment slips | Corpus A needs only tens of adjudications (in-house acceptable). Corpus B's 300–500 needs external annotators — **start W0.** |

---

## Open decisions

1. ~~**Both corpora, or A only?**~~ **Resolved: both. Corpus B is mandatory** — see the notice above. The remaining question is only its scale.
2. **Corpus B scale** — 25 packs with full isolation and annotation (safe, sufficient) vs. 50–100 (stronger, ~2–4× dependency pain). *Recommendation: start at 25 across both strata, expand only if week-1 extraction goes smoothly.*
3. **Annotators** — *Recommendation: external for Corpus B; in-house acceptable for Corpus A's small adjudication set provided the guideline and raw outputs are published.*
4. **Move the Kornia-vs-OpenCV CLAHE case into the introduction?** *Recommendation: yes* — it replaces an unverified statistic with a concrete, verifiable, real-world instance of the exact failure the paper addresses.

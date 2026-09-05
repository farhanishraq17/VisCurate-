# Adjudication of the GPT Critique

**Source:** [`GPT_OPINION.md`](GPT_OPINION.md) (2026-08-18)
**Adjudicated:** 2026-08-18
**Outcome:** 34 points assessed — **28 accepted**, 4 accepted with modification, 2 rejected. All accepted points are reflected in the revised experiment files.

The critique is high quality. It found one manuscript-level correctness bug, one mathematically invalid design in A3, and a cluster of real statistical-validity problems. It also over-reaches in two places. Adjudication below; **§1 is the most important item in this document.**

---

## 1. ★ The merge-policy contradiction — ACCEPTED, and it is worse than reported

GPT identified that `SEMANTIC-PRESERVING` is described inconsistently. Verified against source — there are **four** mutually inconsistent statements, not two:

| # | Location | Statement |
|---|---|---|
| 1 | `sec/appendix.tex:52` | *"`SEMANTIC-PRESERVING` licenses `parameterize` **or `merge`**"* |
| 2 | `sec/6_methodology.tex:125` | *"licenses parameterization or **unification** rather than an outright merge"* |
| 3 | `sec/7_results.tex:60` | *"`SEMANTIC-PRESERVING` licenses **only parameterization**"* |
| 4 | `sec/6_methodology.tex:160` | *"**Because only `EXACT` and `PERCEPTUAL` license a merge**, the binary *mergeable* decision is the axis on which every method is directly comparable."* |

**Statement 4 is the one GPT missed, and it is the dangerous one.** The entire comparability argument of the equivalence-verification benchmark — the basis on which every method in Table 2 is compared — rests on the premise that only `EXACT` and `PERCEPTUAL` license a merge. If `sec/appendix.tex:52` is correct, then:

- the binary `mergeable` positive class used throughout Table 2 is **defined wrongly**
- the headline `0/926` and `0/6` false-merge figures are computed against the **wrong positive class**
- and since `SEMANTIC-PRESERVING` has precision **0.037** with a high false-positive rate on truly-`DISTINCT` pairs, a semantic-licensed merge policy would produce a false-merge rate nowhere near zero

The most likely reality is that the implementation merges only on `EXACT`/`PERCEPTUAL` and `sec/appendix.tex:52` is a documentation error. **That must be verified against the gate implementation, not assumed.** If the code does license semantic merges, the safety results are invalid as reported.

### Required action (blocking, before any new experiment)
1. **Read the gate implementation** and determine the actual policy.
2. Fix all four statements to agree.
3. Recommended policy: **only `EXACT` and `PERCEPTUAL` license `merge`**; `SEMANTIC-PRESERVING` licenses `parameterize` only, never a merge. This is the only policy consistent with precision 0.037 being defensible as cost-benign.
4. Re-run every safety metric under the confirmed policy and state the policy explicitly in the results section.

→ recorded in [`README.md`](README.md) blockers and [`A2/02_verifier_self_validation.md`](A2/02_verifier_self_validation.md)

---

## 2. ★ Pooled MMD destroys conditioning — ACCEPTED, fatal as written

GPT: *"The proposed signature pools different inputs and seeds into one set. This loses conditioning on the probe."*

**Correct, and this was a genuine mathematical error in the original A3 design.**

The original formulation pooled all probes and seeds into one sample set per skill, then took MMD between two pooled sets. Counterexample: skill A outputs `blur(x₁), sharpen(x₂)`; skill B outputs `sharpen(x₁), blur(x₂)`. The pooled output distributions are **identical**, MMD ≈ 0, and the pair certifies as equivalent — while their per-image behavior is completely different. **A false merge by construction.**

The fix GPT proposes is right and has a further merit they did not note: computing a conditional discrepancy per `(x, θ)` and aggregating by worst case is **exactly what the deterministic path already does** (`sec/6_methodology.tex:77`, max over the grid). The corrected design is therefore *more* consistent with the existing method, not less.

→ [`A3/02_method_distributional_equivalence.md`](A3/02_method_distributional_equivalence.md) rewritten

---

## 3. ★ "Zero false merges" needs intervals — ACCEPTED, with a concrete target GPT did not supply

GPT is right that `0/6` is very weak evidence. Quantifying it (Clopper–Pearson, 95%):

| Result | Exact 95% upper bound |
|---|---|
| `0/926` on all distinct pairs | **0.32%** — genuinely strong |
| `0/6` on hard negatives | **39.3%** — nearly vacuous |

The paper designates the hard-negative rate as *"our headline number"* (`sec/6_methodology.tex:192`). **The headline number is the statistically weakest one in the paper.** A reviewer computing that bound sinks the safety claim.

**Concrete target GPT did not provide:** to claim a hard-negative false-merge rate ≤5% at 95% confidence with zero observed errors requires **n ≥ 59** hard negatives. The current slice has 6. This is the single most actionable number in the whole adjudication.

Sources for more hard negatives: engineer additional pairs into `L_0`, plus real ones from A1 Corpus A (cross-library near-misses are natural hard negatives) and A3 (depth-vs-disparity, instance-vs-semantic).

→ [`README.md`](README.md) statistical rules; [`A2/02`](A2/02_verifier_self_validation.md); [`A1/03`](A1/03_annotation_and_metrics.md)

---

## 4. Calibration target ≠ reported safety metric — ACCEPTED

GPT: *"'Precision on non-equivalence' is not automatically the same as controlling false merges."*

Correct and subtle. The paper calibrates on precision with `DISTINCT` as the positive class (`sec/6_methodology.tex:169`) but headlines the false-merge rate on distinct pairs (`sec/6_methodology.tex:192`). These are different quantities, and with 926 distinct vs. ~18 non-distinct pairs, precision on non-equivalence is *inflated by class imbalance* — predicting `DISTINCT` for everything already yields 0.981.

The quantity that actually controls silent merges is the **false discovery rate on the merge decision**, which is never reported.

**Action:** report all three — FPR on distinct pairs, **FDR on the merge decision**, and precision on non-equivalence — so no quantity is hidden behind another. Calibrate against the merge-decision FDR, since that is the safety property claimed.

→ [`A2/01_baseline_ladder.md`](A2/01_baseline_ladder.md) §4

---

## 5. Mapping tables are not output-equivalence claims — ACCEPTED

GPT: *"A named API counterpart or 'same broad operation' may be a migration/category mapping rather than a claim of output equivalence at aligned parameters."*

**Correct, and this was the weakest construct in the original A1 plan.** The original framing measured an "over-claim rate" against maintainers — accusing them of asserting something they never asserted. A reviewer would catch that immediately, and it would damage credibility precisely where the plan was trying to build it.

**Corrected framing:** the tables encode *category-level equivalence judgments of exactly the kind text-based curation produces* — "these two do the same thing." That is the judgment an LLM curator reading descriptions makes. The experiment therefore asks: **when a category-level equivalence judgment is tested by execution at aligned parameters, how often does it hold?** Honest, still makes the point, and no longer misattributes a claim.

The value of the external authorship survives the reframe: the judgments still were not constructed by us.

→ [`A1/01_corpus_A_cross_library.md`](A1/01_corpus_A_cross_library.md) rewritten

---

## 6. Functional-core substitution breaks the construct — ACCEPTED

GPT: *"The published mappings concern transform APIs… Replacing them with deterministic internal functions may sever the connection to the external claim."*

Correct. The mapping tables pair *transform classes* (including sampling behavior, probability, parameter distributions); the plan tested *functional cores*. The thing mapped ≠ the thing tested.

**Resolution adopted:** use the mapping table as a **candidate-pair generator**, and state plainly that the equivalence hypothesis being tested at the functional level is **ours**, not the maintainers'. Where a wrapper is deterministic given a seed, additionally test the wrapper as a secondary check and report both.

→ [`A1/01`](A1/01_corpus_A_cross_library.md)

---

## 7. "A dash is not a pair" — ACCEPTED

Correct and clean. A `–` row means "no built-in counterpart listed" — it names no second function, so "missed consolidation rate" as originally defined was ill-posed. Computing it requires a candidate-generation protocol that searches the other library for behavioral matches. That is a legitimate experiment but a *different* one, and it must be specified rather than assumed.

→ [`A1/01`](A1/01_corpus_A_cross_library.md) §6

---

## 8. Do not clamp or reshape contract-violating outputs — ACCEPTED

GPT: *"Clamping values outside `[0,1]` or guessing BCHW/BHWC can turn a real incompatibility into apparent equivalence."*

Correct, and the original plan explicitly prescribed both. This is a **false-merge generator**: two nodes differing in range handling look identical after clamping.

**Corrected:** contract violations fail the compatibility gate by default. Any repair must be explicit, deterministic, logged, and reported with a sensitivity analysis showing results with and without it.

→ [`A1/02_corpus_B_comfyui.md`](A1/02_corpus_B_comfyui.md) §5

---

## 9. Per-pack containers, not a shared environment — ACCEPTED, reverses the original recommendation

GPT: *"A shared environment creates order-dependent dependency conflicts and makes exclusions partly artifacts of the harness."*

Correct, and it overturns what the plan recommended. The exclusion funnel is a **headline result** — if exclusions are partly artifacts of our environment-resolution order, the result is not credible.

**Corrected:** isolated pinned environment per pack for the primary estimate; the shared environment becomes a clearly-labelled secondary "deployability stress test."

→ [`A1/02`](A1/02_corpus_B_comfyui.md) §4.3

---

## 10. Corpus B is popularity-weighted, not representative — ACCEPTED

Top-installed packs answer *"what affects common deployments"*; they do not estimate ecosystem-wide redundancy. Two different estimands were being conflated.

**Corrected:** stratified design — a popularity stratum plus a reproducible random/long-tail stratum, estimates reported **separately**, with no generalization from the popular stratum to the ecosystem.

→ [`A1/02`](A1/02_corpus_B_comfyui.md) §2

---

## 11. Perceptual equivalence is not transitive — ACCEPTED

Sharp catch. "Compression potential" was defined as the reduction from applying all certified merges, which implicitly takes connected components of the merge graph. But `PERCEPTUAL` equivalence (LPIPS < τ) is **not transitive**: A≈B and B≈C does not give A≈C. Connected-component compression could merge genuinely non-equivalent skills.

**Corrected:** simulate the actual sequential curation policy with re-verification after each proposed merge, and report the transitivity-violation rate as its own finding.

→ [`A1/02`](A1/02_corpus_B_comfyui.md) §7

---

## 12. Annotate agreements, not only disagreements — ACCEPTED

Adjudicating only disagreements yields the over-claim rate but **cannot** estimate accuracy, false-split rate, or relation recall. Those need probability sampling of agreements too, with known inclusion probabilities.

Also accepted: **cluster uncertainty by pack/skill**, since pairs sharing a skill are dependent and treating them as independent understates variance.

→ [`A1/03_annotation_and_metrics.md`](A1/03_annotation_and_metrics.md)

---

## 13. Pruned-pair audit needs power — ACCEPTED

~80 sampled pruned pairs cannot demonstrate high candidate recall across a very large pair universe. **Corrected:** injected known positives + exhaustive evaluation within manageable strata + a powered random audit, reported as an interval.

Related, and **also accepted** (GPT §A4-5): candidate recall measured on `G_0`/`G_ρ` is partly **guaranteed by construction**, because candidate generation always includes same-family pairs and engineered hard negatives (`sec/appendix.tex:43`). Recall must additionally be measured on **real adjudicated positives** from A1.

---

## 14. Do not pre-commit to the "ladder" result — ACCEPTED

The framing "a ladder where the gap narrows but never closes" is an anticipated result presented as a design principle. **Corrected to hypothesis-first:** state the hypothesis, define in advance what outcome would refute it, and permit a baseline to tie or win.

The original did contain an "if a baseline ties, pivot honestly" clause, so this is a strengthening rather than a reversal — but GPT is right that the framing led.

→ [`A2/01`](A2/01_baseline_ladder.md)

---

## 15. Sweep the *existing* baselines too — ACCEPTED

The plan swept new baselines and matched their operating points, while leaving `name-match` at 0.5 and TF–IDF at 0.6 fixed. That is an unfair comparison — in the direction that flatters our result. All threshold-based baselines must be swept and matched identically.

---

## 16. Unparseable LLM replies → `DISTINCT` — ACCEPTED

Mapping unparseable output to `DISTINCT` hides the model's failure rate inside a safety number. **Corrected:** record `ABSTAIN`/`INVALID` explicitly, report its rate, and evaluate coverage–risk tradeoffs.

Also accepted: **"deterministic decoding" and "three seeds" were internally inconsistent** in the original. At temperature 0 there is no seed to vary; nondeterminism is measured by repeated identical calls unless the API exposes real seed control.

---

## 17. AST literal canonicalization rigs the baseline — ACCEPTED

Canonicalizing literals to type placeholders makes the AST baseline blind to **defect type (v), parameter-default drift** — by construction. That is rigging a baseline to fail, the mirror image of the strawman problem the plan was trying to avoid.

**Corrected:** report both a structure-only variant and a semantics-preserving variant that retains defaults, called APIs, and literal values.

---

## 18. Verify `G_ρ`'s answer key is complete — ACCEPTED

The corruption log records injected donor/target relations but may not label every *incidental* relation created among all pairs. Treating unlabeled background pairs as `DISTINCT` by default would manufacture false ground truth.

**Action:** verify completeness before reporting `G_ρ` metrics. This gates the week-0 free re-analysis — do not skip it because the re-analysis is cheap.

Also accepted: **bootstrap by corruption instance / base skill / seed**, not by individual pair.

---

## 19. `COMPLEMENTARY` needs a formal unit of analysis — ACCEPTED

The proposed probe pairs (blur+sharpen, rotate θ / rotate −θ, resize up/down) are **lossy** and only approximately compose. And complementarity is a property of a composition — a triple `(a, b, target)` — not of a pair alone. **Corrected:** define the composition operator, order, target behavior, and tolerance before reporting any F1.

---

## 20. No test-driven threshold repair — ACCEPTED

The proposed `τ_sm` fixes (pre-filter, margin guard, re-calibration) were motivated by a bad test result. Selecting them on the test set is leakage. **Corrected:** select on a fresh validation split, evaluate once on untouched families.

---

## 21. Probe selection must be out-of-sample — ACCEPTED

Greedy diversity-maximizing probe selection evaluated on the same pairs leaks. **Corrected:** select on calibration families, test on disjoint families.

Also accepted: **the 256-probe sweep point exceeds the 177-probe battery** and had no stated source. Either drop it or specify how the battery is extended.

---

## 22. One-at-a-time threshold sweeps miss interactions — ACCEPTED
Thresholds interact in a hierarchical cascade (`τ_pc` controls what reaches the semantic stage). Add a joint sensitivity study.

---

## 23. OOD flips are one-directional evidence — ACCEPTED

Sharp epistemics point. Equivalence is universally quantified over the battery, so a divergence on `B_ood` **refutes** a certificate — but a pair that looked `DISTINCT` on `B` and equivalent on `B_ood` is **not** evidence of equivalence, because the original divergence still stands. The proposed "reverse direction" metric was logically confused and has been removed.

Also accepted: if the battery is extended after seeing OOD failures, the first study is **discovery** and the revised battery must be evaluated on a fresh untouched holdout.

---

## 24. Audit hooks are observability, not containment — ACCEPTED

Important practically, not just for the paper. `sys.addaudithook` **observes**; it does not **contain**. Deliberately executing trojan variants requires OS-level isolation, denied network and credentials, ephemeral filesystems, and resource limits, under a documented threat model. The original conflated the two by describing it as a sandbox extension.

---

## 25–31. A3 statistical corrections — ALL ACCEPTED

| # | Point | Status |
|---|---|---|
| 25 | `K=1` does not algebraically recover the deterministic path (unbiased MMD has `K(K−1)` denominators, undefined at 1) | Accepted — explicit deterministic branch, no identity claim |
| 26 | Per-pair median-heuristic bandwidth prevents a common global threshold | Accepted — fit and freeze per modality on calibration |
| 27 | Naive BCa bootstrap on an MMD U-statistic is invalid near equality (degenerate null, estimator can be negative) | Accepted — permutation / wild bootstrap, coverage validated in simulation |
| 28 | Self-comparison must use **independent** seed batches | Accepted |
| 29 | 8 operators / 30–50 pairs cannot support a 0.99 precision guarantee | Accepted — **A3 reframed as a scoped feasibility study** |
| 30 | Designed relations are hypotheses, not ground truth | Accepted — frozen as hypotheses, adjudicated by experts |
| 31 | `K=32` is a reference, not truth | Accepted — repeat and quantify its own uncertainty |

---

## 32. A4 — eager vs. lazy features is internally inconsistent — ACCEPTED

Sharp catch of a contradiction within the original plan. If learned features are **precomputed for every skill** (the caching argument), the forward cost is already paid and stage short-circuiting saves only distance computation. If features are **lazy**, short-circuiting saves real compute but the cost is no longer simply `O(n)`. The plan asserted both benefits simultaneously. **Corrected:** pick one, document which the implementation does, and make the complexity equation and telemetry match it.

Also accepted: **feature-cache memory was omitted entirely.** At `n=10⁴` with 177 probes × 5 grid points × 768-dim DINO features in float32, the cache is ≈ **27 GB** for DINO alone — closer to ~80 GB with LPIPS and CLIP, and ~800 GB at `n=10⁵`. Memory, not compute, may be the binding constraint at scale, and the original analysis never mentioned it.

Also accepted: **cost per repair must be computed from totals**, not as a mean of per-episode ratios, with a defined behavior for models producing zero repairs (Llama 3.2 1B scores 0.000 on everything).

Also accepted: GPU timing needs a real benchmark protocol (warm-up, CUDA event sync, repeated trials, cold/warm, median **and tail**, intervals).

---

## 33. Tie-break scoring needs a policy — ACCEPTED
Runtime, dependencies, memory, and tests are incomparable objectives; a weighted sum is arbitrary. **Corrected:** documented lexicographic rule or a Pareto frontier with stated priorities, compared against meaningful named policies (fastest, fewest dependencies, best-tested) rather than an arbitrary baseline.

---

# Rejected and modified

## R1. "The folder contains plans, not evidence" — REJECTED as framing, ACCEPTED as substance

The task was to produce experiment plans. Criticizing planning documents for not being results is a category error, and the scorecard's "Current state: not run" columns are not defects of the plans.

**However the underlying substance is right and has been adopted:** each experiment now specifies the artifact set it must produce — frozen manifests with hashes, locked environments, raw per-unit outputs, exclusion logs, machine-readable metrics with denominators and intervals, run manifests with commit SHA and hardware, and scripts that regenerate every table and figure. That requirement was genuinely missing.

## R2. Depth affine alignment "hides real incompatibility" — MODIFIED

Partially right, but the proposed remedy overshoots. Affine alignment is **standard and necessary** for monocular depth — every depth benchmark does it, because depth is defined only up to scale and shift. Without it, Depth Anything vs. MiDaS trivially reads `DISTINCT` and the comparison measures output conventions rather than behavior.

**Adopted as an addition, not a replacement:** keep affine-invariant comparison as the behavioral metric, and report native resolution and calibration convention as **separate interface-compatibility attributes**. GPT's underlying concern — that alignment can erase user-relevant behavior — is handled by reporting both rather than by abandoning alignment.

## R3. Execution order: A2 entirely before A1 — MODIFIED

Largely agreed, and the plan already put A2's free `G_ρ` re-analysis in week 0. But the ordering rationale differs: **A2 goes first because it is nearly free, not because it matters more.** A1 remains the highest-value deliverable — it is the AC's item #1 and the only weakness all three reviewers raised. A1 Corpus B must not be starved to fund optional A2 breadth, and GPT's own A1 verdict agrees that Corpus B is *"the clearest direct answer."*

## R4. Corpus A may not satisfy the AC's "uncurated" requirement — ACCEPTED, and it changes the plan materially

Not a rejection — flagged separately because it **overturns a recommendation**. OpenCV, Kornia, and Albumentations are each individually *curated*, well-maintained libraries. Corpus A tests cross-library equivalence judgments, which is valuable, but it is **not "an uncurated open-source skill library."**

The original plan offered Corpus A as a sufficient fallback if Corpus B slipped. **That was wrong.** Corpus B is the only corpus that literally satisfies AC item #1.

**Corrected:** Corpus B is now **mandatory**, with reduced scale as the contingency rather than omission. Better to audit 25 packs with strong isolation and real annotation than 100 packs fragilely.

→ [`A1/README.md`](A1/README.md)

---

# Items GPT missed

| # | Finding |
|---|---|
| **M1** | The merge-policy contradiction also invalidates the **comparability argument** at `sec/6_methodology.tex:160`, which explicitly justifies the binary mergeable axis by *"because only `EXACT` and `PERCEPTUAL` license a merge."* See §1. |
| **M2** | A concrete sample-size target for the safety claim: **n ≥ 59** hard negatives for a ≤5% upper bound at 95% confidence with zero observed errors. Current slice: 6. |
| **M3** | Feature-cache **memory** at scale (~27 GB DINO-only at `n=10⁴`) was absent from both the plan and the critique. See §32. |
| **M4** | R1's *"battery of over 200 probe images"* is a **misreading** — the battery is 177 probes; the 200 is the action-step budget (`sec/6_methodology.tex:151`). The revision should separate these prominently so the misreading is not repeated in round 2. |

---

# Net effect on the plan

| Change | Impact |
|---|---|
| Merge-policy contradiction resolved | **Blocking** — gates all safety metrics |
| A3 math rewritten (conditional, not pooled) | Prevents a false-merge-by-construction design |
| Hard-negative slice expanded 6 → ≥59 | Makes the headline safety claim statistically meaningful |
| Corpus B promoted to mandatory | Correctly aligns A1 with AC item #1 |
| Corpus A construct reframed | Removes a misattribution a reviewer would catch |
| Per-pack isolation; no output clamping | Prevents harness artifacts contaminating headline results |
| Artifact/reproducibility requirements added | Makes results defensible under replication |
| A3 rescoped to feasibility study | Honest given available sample support |

**Scope verdict:** GPT is right that the suite is over-scoped. The mandatory core is now explicitly: **merge-policy fix → A4 telemetry → A2 core (`G_ρ` replay + fair baselines) → A1 Corpus A + reduced Corpus B → A4 measured cost → A3 feasibility.** Everything else is labelled optional in the revised files.

# GPT Opinion: Independent Critique of the Phase-2 Experiment Suite

**Review date:** 2026-08-18  
**Scope inspected:** every file in `Experiments/A1` through `A4`, the suite-level README, and the adjacent reviewer, planning, methodology, results, appendix, and discussion documents needed to verify status and consistency.  
**Role:** skeptical reviewer / methodological judge, not co-author.

## Bottom-line verdict

**No: the proposed experiments are not done, and they are not yet in their best form.**

The `Experiments` folder contains 17 Markdown planning documents and no experiment code, frozen manifests, datasets, logs, metric outputs, plots, annotation files, environment lockfiles, or result tables. Every execution checklist remains unchecked. The suite README itself labels A1, A3, and A4 as **Planned** and A2 as only **Partly done in draft**. A search of the surrounding Phase-2 material found the manuscript and planning/reviewer documents, but no implementation or result artifacts for the new A1–A4 deliverables.

The plans are unusually thoughtful and contain many good safeguards: frozen pre-test artifacts, self-pair gates, matched operating points, exclusion logs, real-library annotation, abstention, modality-aware comparison, candidate-recall measurement, and measured rather than asserted scalability. However, the suite is too broad for one revision cycle, sometimes pre-commits to favorable conclusions, and contains several statistical and construct-validity problems that could undermine the paper even if everything were executed exactly as written.

### Overall scorecard

| Item | Current state | Design quality | Judgment |
|---|---|---:|---|
| **A1: real-library evaluation** | Not run | Strong idea, material validity risks | Must do, but redesign parts |
| **A2: baselines + verifier validation** | Three old baselines/tables exist; proposed additions not run | Highest value per unit effort | Do first after instrumentation |
| **A3: neural/stochastic extension** | Not run | Promising but statistically incorrect in current form | Redesign before implementation |
| **A4: scalability/cost** | Not run | Strong decomposition, incomplete measurement protocol | Instrument first; report conservatively |

## Submission-critical findings

### 1. The folder contains plans, not evidence

The documents repeatedly use target language such as `X%`, `Y%`, “expected,” “proves,” and “target sentence.” None of those placeholders is supported by outputs in this folder. Until code, frozen inputs, environment manifests, raw results, and reproducible analysis artifacts exist, the paper cannot claim these experiments were conducted.

At minimum, each experiment needs:

- a frozen input/pair/model manifest with hashes and provenance;
- executable code plus a locked environment/container;
- raw per-unit outputs and failure/exclusion logs;
- a machine-readable metrics table with denominators and uncertainty intervals;
- a run manifest containing commit SHA, hardware, seed, timestamps, and configuration;
- scripts that regenerate every reported table and figure.

### 2. The safety policy contradicts the “cost-benign” defense

The current manuscript appendix says `SEMANTIC-PRESERVING` can license **parameterize or merge**. The results discussion and A2 plan defend its extremely low precision (`0.037`) by saying this relation licenses only parameterization and therefore cannot silently remove a capability. Both cannot be true.

This is a major correctness issue. With precision 0.037, allowing a semantic verdict to authorize merging is unsafe. The policy should be made consistent everywhere: only `EXACT` and, if carefully justified, `PERCEPTUAL` may license an automatic merge; `SEMANTIC-PRESERVING` should retain both implementations or require human review. Re-run all safety metrics under the final policy.

### 3. “Zero false merges” is not “zero risk”

Claims such as `0/926` and especially `0/6` need exact confidence intervals. Observing zero errors among six hard negatives provides very weak evidence; its 95% upper error bound is large. Every false-merge result should include a numerator, denominator, exact interval, and an a-priori sample-size/power target. The same applies to A3: a 30–50-pair set spread across many relations cannot substantiate “safety preserved” unless the number of genuinely distinct/hard-negative cases is large enough.

### 4. The suite is over-scoped

The plans combine a real cross-library audit, a 50–100-pack ComfyUI audit, external annotation, seven additional baselines, verifier replay over 300 instances, multiple ablations, OOD testing, trojan instrumentation, neural distributional equivalence, a model-zoo study, expanded curator APIs, cost modeling, and maintainability tie-breaking. This is closer to a new paper than a revision.

The acceptance-critical package should be: **A4 telemetry hooks → A2 verifier replay and fair baselines → A1 real audit → minimal empirical scalability → reduced A3**. Do not weaken the mandatory items by attempting every optional deliverable.

## A1 critique: real and uncurated libraries

### What is strong

- Testing externally authored cross-library mappings is much stronger than inventing all pairs internally.
- Corpus A is tractable and likely to produce interpretable case studies.
- Corpus B directly addresses real ecosystem decay and gives a realistic exclusion funnel.
- Freezing pair lists before execution, logging canonicalization, adjudicating disagreements blindly, and sampling fingerprint-pruned pairs are excellent choices.

### What must change

1. **Do not overstate mapping tables as ground-truth equivalence claims.** A named API counterpart or “same broad operation” may be a migration/category mapping rather than a claim of output equivalence at aligned parameters. Treat the maintainer label as a noisy external hypothesis and quote its exact semantics. Report results by label type rather than collapsing them into “direct equivalent.”

2. **The functional-core substitution changes the object being tested.** The published mappings concern transform APIs, many of which include sampling behavior, defaults, probability, and parameter distributions. Replacing them with deterministic internal functions may sever the connection to the external claim. Either evaluate the public wrappers distributionally/at controlled seeds, or explicitly call Corpus A an author-constructed functional-API audit rather than a direct audit of the maintainer table.

3. **A dash is not a pair.** A `–` entry means no listed built-in counterpart; it does not identify a concrete alternative function. “Missed consolidation” cannot be computed from such rows until a candidate-generation protocol specifies which real function is being paired and why.

4. **Parameter alignment is a major source of researcher degrees of freedom.** Documentation-only alignment is helpful but insufficient. Pre-register mappings, use two independent reviewers for ambiguous mappings, preserve alternative valid alignments, and report sensitivity across them. Alignment failure and behavioral nonequivalence must be separate outcomes.

5. **Do not silently clamp or heuristically reshape contract-violating outputs.** Clamping values outside `[0,1]` or guessing BCHW/BHWC can turn a real incompatibility into apparent equivalence. Contract violations should normally fail the compatibility gate; any repair/canonicalization must be explicit, deterministic, logged, and evaluated in a sensitivity analysis.

6. **Corpus B is popularity-weighted, not ecosystem-representative.** The top-installed packs answer “what affects common deployments,” not “what exists in the ecosystem.” Use a stratified design: popular packs plus a reproducible random/long-tail sample. Report estimates separately rather than generalizing from the top packs to all ComfyUI.

7. **Use per-pack containers, not one shared environment, for the main estimate.** A shared environment creates order-dependent dependency conflicts and makes exclusions partly artifacts of the harness. A shared environment can be a secondary “deployability stress test,” while isolated pinned environments should support the behavioral estimate.

8. **Annotation only on verifier/maintainer disagreements cannot estimate total verifier accuracy.** It can confirm an over-claim rate, but accuracy, false-split rate, and relation recall require probability sampling of agreements and disagreements with known inclusion probabilities. Cluster uncertainty by pack/skill, not by treating millions of pairs as independent.

9. **The pruned-pair audit needs a power calculation.** Sampling about 80 pruned pairs is too small to demonstrate high candidate recall in an enormous pair universe. Combine known/injected positive pairs, exhaustive evaluation within manageable strata, and a statistically powered random audit. Report an interval, not only a point estimate.

10. **Pairwise perceptual equivalence may not be transitive.** “Compression potential” cannot safely be computed by taking connected components unless the merge relation is proven transitive under the operating rule. Simulate the actual sequential curation policy and re-verify after each proposed merge.

### Verdict on A1

**Essential and potentially the strongest contribution, but Corpus A alone may not fully satisfy the AC request for an “uncurated open-source skill library.”** The cross-library APIs are individually curated, and the relation graph is partly author-constructed. Corpus B, even at reduced scale, is the clearest direct answer. If time is short, use fewer packs with stronger isolation, annotation, and reproducibility rather than a larger but biased and fragile crawl.

## A2 critique: baselines, self-validation, ablations, and scope limits

### What is strong

- Re-analyzing existing `G_rho` instances is the highest-value low-cost task.
- Adding source-code and structural baselines closes an obvious unsupported claim.
- Probe, backend, threshold, seed, OOD, and side-effect studies address the reviewers directly.
- Full curves are better than single arbitrary thresholds.

### What must change

1. **Do not pre-commit to a “ladder where the gap narrows but never closes.”** That is an anticipated result, not a design principle. The analysis must permit a sentence/code/LLM baseline to tie or beat VisCurate. Define hypotheses and evaluation rules before observing outcomes.

2. **Use one consistent target and fair calibration.** If the safety target is automatic merging, define `mergeable` as the positive class and directly control/report precision, false-discovery rate, and recall for that decision. “Precision on non-equivalence” is not automatically the same as controlling false merges. Sweep or calibrate the existing name/TF-IDF baselines too; comparing their fixed arbitrary thresholds with matched thresholds for newer baselines is unfair.

3. **Do not map unparseable LLM answers to `DISTINCT`.** That artificially improves safety while hiding model failure. Record `ABSTAIN/INVALID`, include its rate, and evaluate coverage-risk tradeoffs. “Deterministic decoding” and “three different seeds” also need reconciliation; at temperature zero, use repeated API calls to measure nondeterminism, not pretend seed control exists unless the API exposes it.

4. **AST literal canonicalization removes behaviorally critical information.** Turning all literals into type placeholders will deliberately hide different defaults/constants. Report both structure-only and semantics-preserving AST variants, including defaults, called APIs, and literal values.

5. **Verify that `G_rho` has a complete relation answer key.** A corruption log labels injected donor/target relations, but may not label every incidental relation among all pairs. Do not treat unlabeled background pairs as `DISTINCT` by default. Bootstrap by corruption instance/base skill or seed, not by individual pair, because pairs share skills and are dependent.

6. **The complementary relation needs a formal unit of analysis.** Many listed examples (blur+sharpen, rotate and inverse rotate, resize up/down) are lossy and only approximately compose. Complementarity is usually a relation among a composition/triple, not merely two skills. Define the composition operator, target behavior, order, tolerance, and ground truth before reporting F1.

7. **Avoid test-driven threshold repair.** The weak semantic result can motivate a new method, but any new pre-filter, margin, or threshold must be selected on a fresh validation/calibration split and evaluated once on untouched test families.

8. **Battery selection must be learned out-of-sample.** Greedy diversity selection on the same skill pairs used for evaluation leaks signal. Select probes on calibration families and test on disjoint families. The proposed 256-probe point also needs an explicit source because it exceeds the current 177-probe battery.

9. **One-at-a-time threshold sweeps miss interactions.** Add at least a small joint/global sensitivity study for the thresholds that interact in the hierarchical cascade.

10. **OOD certificate flips are one-directional evidence.** Finding a counterexample can invalidate an equivalence certificate; a pair that appears equivalent on an OOD subset does not undo a previously observed divergence. Remove or carefully reinterpret the proposed `DISTINCT -> equivalent` “reverse” result. If the battery is expanded after seeing OOD failures, label the first study as discovery and evaluate the revised battery on a new untouched holdout.

11. **Audit hooks are observability, not containment.** Intentionally executing trojan variants requires OS-level isolation, denied network/credentials, ephemeral filesystems, strict resource limits, and a documented threat model. `sys.addaudithook` can support telemetry for Python behavior but must not be presented as a security boundary or native-code defense.

### Verdict on A2

**This should be the first completed scientific work package.** The `G_rho` verifier replay, fair baseline curves, support counts, confusion matrix, exact uncertainty intervals, and a leakage-free probe ablation will address more reviewer risk than the optional model/backend breadth.

## A3 critique: distributional equivalence

### What is strong

- Within-model variance first, explicit abstention, modality-specific comparators, sample-size sweeps, and reduced-scope segmentation/depth are sensible.
- Recognizing that stochastic equivalence requires distributions rather than matched individual pixels is correct.
- Keeping generative claims narrow is scientifically mature.

### Fatal issues in the current mathematical design

1. **The proposed signature pools different inputs and seeds into one set.** This loses conditioning on the probe. Two models could swap their behavior across images and still have similar pooled output distributions. Behavioral equivalence requires comparing `P(output | input, parameters)` for each probe/grid point, then aggregating conservatively across inputs. Compute a conditional discrepancy per `(x, theta)` and combine using a predeclared worst-case, quantile, or multiple-testing rule.

2. **`K = 1` does not recover the deterministic method as written.** The unbiased MMD estimator contains `K(K-1)` denominators and is undefined at one sample, and MMD in feature space is not the existing deterministic cascade. Implement an explicit deterministic branch for `K=1`; do not claim algebraic identity.

3. **A per-pair median-heuristic kernel bandwidth prevents a common global threshold.** Distances become pair-adaptive and not directly comparable. Fit bandwidth(s) on the calibration set per modality and freeze them with the thresholds.

4. **Naive bootstrap confidence intervals for MMD U-statistics are delicate.** The unbiased estimator can be negative and has nonstandard behavior under equality. Validate coverage/type-I error in simulation and prefer a statistically appropriate permutation/wild-bootstrap or established kernel equivalence-testing procedure. Merely placing a BCa interval around MMD and calling it TOST is not sufficient.

5. **Self-comparison must use independent draws.** Comparing a sample set with itself trivially understates variance. Use independent seed batches (and repeated batches) for within-model null distributions.

6. **The calibration set is far too small for the promised guarantee.** Eight Tier-1 operators and 30–50 designed pairs cannot support family-disjoint calibration, multi-relation test evaluation, and a claimed 0.99 non-equivalence precision with useful confidence. Either expand independent model families/pairs substantially or frame A3 as a scoped feasibility/case study without a high-confidence safety guarantee.

7. **Designed relations are hypotheses, not ground truth.** Statements such as SAM variants being `PERCEPTUAL` or depth-family pairs being `SEMANTIC-PRESERVING` are contestable. Freeze them as hypotheses, then obtain task-aware expert adjudication and/or dataset ground truth. Do not score P/R/F1 against the authors' expectation.

8. **Depth alignment and mask resizing can hide real incompatibility.** Make comparison symmetric, preserve native resolution as a separate compatibility attribute, and report both convention-invariant task quality and interface-level differences. Per-pair affine fitting can erase calibration behavior that matters to users.

9. **A `K=32` run is a reference, not truth.** Repeat the high-K reference and quantify its own uncertainty. For high-dimensional generative features, `K=8–16` is likely underpowered; allow “not practically certifiable” as the result.

### Verdict on A3

**Do not implement the present pooled-MMD formula.** Redesign it as conditional, per-probe distributional comparison with validated inference. For the revision, restrict the empirical claim to two modalities (segmentation and depth), use enough independently labeled pairs, and report a feasibility boundary rather than claiming the classical safety property automatically transfers.

## A4 critique: scalability, cost, and tie-breaking

### What is strong

- Separating signature computation, candidate generation, and cached pair comparison is the right analytical decomposition.
- Candidate reduction without recall is correctly recognized as meaningless.
- Per-stage timing, model-cost Pareto plots, and real-node runtime tails would be useful to practitioners.

### What must change

1. **State complexity precisely.** Forward execution can be `O(n)` in the number of skills, but pairwise cached comparisons can still be quadratic, feature storage is `O(n * probes * grid * feature_dim)`, and candidate density may grow with `n`. Do not summarize the entire method as “the expensive part is O(n)” without measured memory and candidate-growth behavior.

2. **Reconcile caching with short-circuit savings.** If LPIPS/DINO/CLIP features are precomputed for every skill, learned forward cost is already paid before pair-stage short-circuiting. If features are lazy, compute depends on which pairs reach each stage. The telemetry and complexity equation must match the actual implementation.

3. **`cache_hit` does not prove asymptotic complexity.** It measures reuse in one workload. Prove the algorithmic bound from the implementation and use telemetry to estimate constants and validate that repeated computation does not occur.

4. **GPU timing needs a real benchmark protocol.** Specify warm-up runs, CUDA synchronization/events, repeated trials, cold vs. warm cache, I/O inclusion, concurrency, power mode, median/tail latency, confidence intervals, and peak-memory measurement. Hardware metadata alone is not enough.

5. **Synthetic candidate recall may be inflated by construction.** The current system always includes same-family pairs and engineered hard negatives, so high recall on `G0/G_rho` can be partly guaranteed. Include real adjudicated positives and report recall by source/family, with uncertainty.

6. **Extrapolation from `n <= 100` to `n = 10,000` is fragile.** Use multiple real/synthetic library sizes, repeated subsamples, fitted candidate-density models, and sensitivity bands. Mark projections prominently and avoid a single-point GPU-hour claim.

7. **Compute cost per repair from totals, not an average-of-ratios shortcut.** Use total cost divided by total correct repairs (with uncertainty), define behavior when a model produces zero repairs, timestamp provider prices, and report tokens/latency separately so dollar results remain reproducible after pricing changes.

8. **The tie-break score needs a policy, not arbitrary weighted mixing.** Runtime, dependencies, memory, tests, and maintainability are incomparable objectives. Prefer a documented lexicographic rule or Pareto frontier with user-selected priorities, and report sensitivity. Compare against meaningful policies (fastest, smallest dependency footprint, best-tested), not only an arbitrary choice.

### Verdict on A4

**Build the telemetry before any new run, but keep the first paper result modest:** measured end-to-end cost at several actual `n`, candidate recall/reduction with intervals, memory, and a transparent projection range. Curator-roster expansion and a composite maintainability score are optional.

## Recommended execution order and minimum publishable package

1. **Fix manuscript/policy blockers immediately:** the semantic-merge contradiction, missing Intrinsic Score definition, empty Training Details, broken LaTeX environment, rebuttal template, citation verification, and inconsistent probe count (`177` vs. “200+”).
2. **Implement A4 telemetry** with a proper timing protocol and immutable run manifests.
3. **Complete A2 core:** replay `G_rho`, verify the full answer key, report support/confusion/intervals, run all baselines under a fair binary merge target, and perform leakage-free probe-size/class ablations.
4. **Run A1 Corpus A with corrected construct claims**, pre-registered parameter mappings, and blinded adjudication that samples agreements as well as disagreements.
5. **Run a reduced but genuine Corpus B audit** using isolated pack environments and popularity/long-tail strata. This is the clearest answer to the AC's real-library requirement.
6. **Report A4 measurements** at actual sizes before extrapolating.
7. **Only then implement reduced A3** after correcting the conditional-distribution statistics. If time or sample support is insufficient, explicitly scope A3 as a feasibility study.

## Final judgment

The suite has the ingredients of a strong resubmission, but currently it is **an ambitious research plan, not a completed experimental revision**. Its best qualities are methodological honesty, focus on safety, and direct alignment with the reviewers. Its greatest risks are scope explosion, optimistic target-language before evidence, weak sample support for safety claims, and a mathematically invalid pooling step in A3.

If the team executes fewer experiments with frozen provenance, adequate statistical power, complete artifacts, and honest negative results, the work will be stronger than attempting all T1–T29/F1–F12 deliverables superficially. The decisive evidence is not the number of experiments; it is whether the real-library result, verifier validation, and scalability claim remain credible under skeptical replication.

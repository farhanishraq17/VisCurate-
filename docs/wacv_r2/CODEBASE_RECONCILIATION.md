# Reconciliation Against the Real VisCurate Codebase

> **★ UPDATE 2026-08-19 — read this before anything below.** §1's "find out where the paper's
> numbers came from" question is now answered, from a newer repo the user pointed to:
> **https://github.com/farhanishraq17/VisCurate-** (pushed 2026-07-05, well past the local
> snapshot's last commit). It resolves the *sizing* discrepancy (944 vs. 24 — a misreading on my
> part, see §7) but surfaces a **more serious problem**: the repo contains two documents from the
> same push that **directly contradict each other** about whether the paper's headline numbers
> are real. See new §7 below. The original §1–§6 are left as written for the record; §7 is the
> current, superseding finding.

---

**Source:** `D:\Research_Ishmam\Viscurate\Serious Deadlines\VisCurate_Experiments\` (commit `00f2d2c`, "completed phases 4-9")
**Read:** `claude.md` (roadmap/spec, 639 lines), `docs/phase_summaries.md` (Phases 0–9, 715 lines), `PROJECT_STATUS_REPORT.md`, `Project_Pipeline.md` headers, plus direct reads of `curation/gating.py`, `equivalence/relations.py`, `baselines/judges.py`, `experiments/audit.py`, `equivalence/candidates.py`, `corruption/grid.py`, `cli.py`.
**Reconciled:** 2026-08-18

> **Note on location.** The codebase lives at `Viscurate/Serious Deadlines/VisCurate_Experiments/`. This `Experiments/` planning folder lives at `Viscurate/Phase - 2/Experiments/`, and the paper source is at `Viscurate/Phase - 2/_WACV_27_.../`. These are three separate directories. Nothing here assumes they get merged — but whoever implements A1–A4 needs to know the real code is not next to the plans.

---

## 1. The one finding that matters most — read this first

**The numbers currently sitting in the paper draft do not trace to this codebase.**

The paper (`sec/6_methodology.tex:141`) claims:
> *"`G_0` enumerates $944$ designed ordered pairs (seed `1234`)... $713$ fit pairs and $100$ leakage-check pairs."*

The actual `configs/ground_truth_g0.yaml` in the codebase is **80 lines**, and the codebase's own phase summary states plainly:

> *"24 designed pairs (7 subsumption, 10 semantic, 1 complementary, 6 DISTINCT hard negatives); everything else defaults to DISTINCT."*

And Table 2's headline `0/926`, Table 3's per-relation F1s, and the entire Table 1 model-evaluation grid (GPT-5.5 at μF1 0.583, etc.) — none of these exist anywhere in this repo. `PROJECT_STATUS_REPORT.md` says it without hedging:

> *"There is no `results/` directory, no CSVs, no figures, and zero paper numbers anywhere in the repo... the real LPIPS/DINO/CLIP divergence run over the full battery... has not been run."*

**This is not a contradiction I'm alleging — it's what the codebase's own status report says about itself.** Three possibilities, in order of likelihood:

1. A different, later version of this codebase (not present on this machine) produced the paper's numbers — plausible, but then that version needs to be located before anything else happens, since it's the actual source of truth for what's already been measured.
2. The paper's numbers are illustrative placeholders that were never meant to ship — which would need to be corrected before any submission, independent of everything else in `Experiments/`.
3. The numbers came from manual/exploratory runs that were never saved as artifacts — recoverable, but they need to be re-run and captured properly this time.

**Action before anything else in this plan:** find out which of these it is. If it's (1), locate that codebase version. If (2) or (3), the paper's current Tables 1–3 need to be treated as **not yet real** and re-run from this codebase — which, favorably, is exactly what A2's plan already prescribes. This isn't extra work A2 didn't anticipate; it just means A2 is now load-bearing for the paper's *existing* numbers, not only its *new* ones.

I'm flagging this at the top because it changes what "ready" means: the GPU run isn't only about the four AC revision items anymore — it's also about making the paper's current results real for the first time.

---

## 2. Second finding — the hardened sandbox does not exist yet

`curation/sandbox.py` implements `ExecutionPolicy`, which **blocks** any `trusted=False` skill from executing. This is deliberate and documented as unimplemented-by-design:

> *"The hardened sandbox is deliberately not implemented... `allow_untrusted` MUST stay `False` outside a reviewed sandbox... No untrusted execution is implemented or enabled."*

**This blocks two things in the `Experiments/` plans directly:**

| Plan item | Why it's blocked |
|---|---|
| **A1 Corpus B** — executing ~1,000 real ComfyUI community nodes | Every one of those nodes is untrusted third-party code by the codebase's own trust model |
| **A2's `trojan` defect type** — deliberately executing skills with malicious side effects | Same boundary, higher stakes |

**This is not a research-design problem — my A1/A2 methodology for these is unaffected — it's an engineering prerequisite that has to land before either can run at all.** The codebase's own `HARDENING_PLAN` document (referenced in `sandbox.py`'s docstring, not yet read here) presumably specifies network namespace, restricted FS, rlimits+timeout, no `eval`/`exec` of skill source — matching almost exactly what [`A2/04_ood_and_scope_limits.md`](A2/04_ood_and_scope_limits.md) §2.3 already specified as the containment/observability split. **That section's design is compatible with what the codebase already anticipates building; it just hasn't been built.**

**Sequencing implication:** hardened-sandbox implementation moves in front of A1 Corpus B and A2's trojan work in the execution order.

---

## 3. What's already built vs. genuinely new, per experiment

### A1 — Real library audit

| Plan item | Codebase reality |
|---|---|
| Canonicalization contract (dtype/range/channel/shape rules) for real-world skills | **Exists**: `skills/canonicalize.py`, versioned (`CANON_VERSION`), handles dtype/range coercion, 1↔3 channel, RGBA-over-midgray + separate alpha tracking, shape gate, binary-mask IoU path. **Reuse this contract's design for Corpus A/B adapters instead of inventing a new one** — same rules, just need an adapter layer mapping ComfyUI's `BHWC float32 [0,1]` and each classical library's convention onto it. |
| Parameter alignment maps (hand-authored, cross-library) | **Partially exists**: `equivalence/param_alignment.py` + `configs/param_alignment.yaml` already implement the *concept* — symmetric matched-sweep axes and asymmetric subsumption grids, "never hard-coded." It's built for *within-library* skill pairs (e.g. `blur_gaussian`/`blur_box`), not cross-library (OpenCV vs. Kornia vs. PIL). **The mechanism transfers directly; the maps themselves are new** — author Corpus A's alignment maps in this same YAML format so they run through the existing `taxonomy.classify` unchanged. |
| Hardened sandbox for untrusted third-party code (Corpus B) | **Does not exist** — see §2. Blocks Corpus B until built. |
| Annotation protocol + κ computation | **Exists**: `benchmark/human_review.py` extracts a labeling slice to a JSON template, loads completed annotations, computes Cohen's/Fleiss' κ, and reports `status="pending"` honestly when no annotations exist yet. **Reuse this directly** for A1's Corpus A/B annotation rather than building a new pipeline — it already does exactly what [`A1/03_annotation_and_metrics.md`](A1/03_annotation_and_metrics.md) specifies. |
| Realism/reproducibility audit | I originally guessed `experiments/audit.py` might be a real-library auditor — **it is not**. It's an internal QA check (probe license coverage, split disjointness, corruption-grid completeness, artifact presence). Not reusable for A1; A1's real-library audit is a genuinely new capability. |

### A2 — Baselines + verifier self-validation

| Plan item | Codebase reality |
|---|---|
| Rungs 1/2/6 (name-match, TF-IDF/embedding-cosine, LLM-on-descriptions) | **Fully built**: `baselines/judges.py` — `NameMatchJudge`, `EmbeddingCosineJudge` (swappable `TextEmbedder` protocol, TF-IDF is the default), `LlmJudge` (behind `LlmClient` protocol; records "not run" rather than fabricating with no client). **Do not rebuild these.** Rungs 3 (sentence-embedding), 4 (code-embedding), 5 (AST clone), 7 (LLM-on-source) are genuinely absent — build them as new `TextJudge`/new-protocol implementations in the same package, following the existing pattern. |
| Embedding-cosine unfair fixed threshold | **Already flagged as a known issue by the codebase itself**: *"A single fixed τ on the full-corpus TF-IDF can be conservative... the fair comparison — the text baseline at its own best operating point (or a swept ROC) — is a reporting choice, not a re-architecture. Recommended for the real run."* This is precisely the fix demanded in [`A2/01_baseline_ladder.md`](A2/01_baseline_ladder.md) §4.2 — already anticipated, not yet done. |
| `G_ρ` merge-recall re-analysis | **The runner already exists**: `benchmark/runner.py` computes candidate pairs → per-track verdicts → full per-pair distances → metrics, device-parameterized. It currently scores against `G_0` (24 pairs). Running it against the corrupted `G_ρ` graphs (Phase 5's output) for the merge-recall fix in [`A2/02_verifier_self_validation.md`](A2/02_verifier_self_validation.md) is a **configuration change to an existing runner**, not new code. |
| Metrics: per-relation P/R/F1, 6×6 confusion, mergeable decision, false-merge/precision-on-DISTINCT, abstention rate, divergence-by-true-relation | **All exist**: `benchmark/metrics.py`. This is every metric [`A2/02`](A2/02_verifier_self_validation.md) asks for except the hard-negative-count power calculation (n≥59), which is a *design* input (build more hard negatives) not a *metric* gap. |
| Calibration protocol, cluster-disjoint split | **Exists**: `equivalence/calibrate.py` — `select_threshold` (recall-subject-to-precision-floor), `calibrate_from_result` (cluster-disjoint split, provenance-stamped). Matches [`A2/02`](A2/02_verifier_self_validation.md)'s "no test-driven repair" rule exactly — the codebase already enforces validation/test separation structurally. |
| `COMPLEMENTARY` formal definition | **Already implemented, not "undefined" as I assumed**: `equivalence/complementary.py` — non-triviality + approximate commutation `D(A(B(x)), B(A(x)))` small, executed on real compositions. The codebase's own docs already flag the exact caveat I raised independently: *"necessary, not sufficient... two linear filters commute yet are not 'disjoint aspects'; caught earlier (SEMANTIC) in the full pipeline."* **My §19 correction in `A2/02` (formalize the composition operator, order, target, tolerance) should point at this existing module and its known limitation, not propose building a detector from scratch.** |
| Intrinsic Score (IS) — flagged in `PLAN.md` as **undefined in the paper** | **Fully defined in code**: `studies/metrics.py::intrinsic_curation_score()` — *"ideal-action F1 with a penalty for rejected/blocked/invalid budget-spending actions."* This is a **paper-text fix, not a research gap** — copy the real definition from code into `sec/7_results.tex:23-25` where it currently reads "defined as \\" and stops. |
| `sys.addaudithook` / trojan side-effect channel | Depends on the hardened sandbox (§2) — not yet buildable. |

### A3 — Neural/stochastic extension

| Plan item | Codebase reality |
|---|---|
| Seeded-stochastic determinism handling | **Exists, but narrower than A3 needs**: `claude.md` §1.4 and the skill library already classify 4 skills as seeded-stochastic (noise, random-crop, etc.) — deterministic *given a fixed seed*, compared at matched seeds. This is the mechanism A3's method doc (§0) explicitly said the paper's `K=1`-style fixed-seed handling *cannot* be stretched to cover genuinely non-deterministic neural models or models with no shared seed space. **A3's conditional-distributional design is still net-new** — but it should be built as a sibling classification (a fourth determinism category alongside deterministic/seeded-stochastic/precision-sensitive/platform-sensitive) rather than a parallel system, so it inherits the existing dispatch logic in the skill model rather than duplicating it. |
| Comparator backends (LPIPS/DINO/CLIP) | **Exist and match exactly**: `equivalence/backends.py` — `LpipsBackend` (AlexNet), `DinoBackend` (`vit_base_patch16_224.dino`), `ClipBackend` (`ViT-B-32-quickgelu`, openai), lazy-imported, one-model-at-a-time (6GB GPU discipline), context-manager `close()`. **A3's modality-specific comparators (mask IoU, depth AbsRel, etc.) are new**, but they plug into the same `PerceptualBackend`/`SemanticBackend` protocol pattern rather than inventing a new backend interface. |
| `ComparatorView` / `OutputProvider` type-enforced modality boundary | **Directly reusable and important**: neural skills wrapped for A3 must implement the same `Skill.comparator_view()` contract (no `description` attribute) — this is how the text-blindness guarantee is structurally enforced everywhere else, and A3 should not create a side channel that bypasses it. |
| Distributional/statistical inference (permutation nulls, TOST, bootstrap validation) | **Genuinely new** — nothing in the codebase does distributional comparison; everything is point-valued today. This is the real, substantial new engineering A3 requires. |
| Generative/neural skills in scope at all | **Explicitly out of v1** per `claude.md`: *"No generative/diffusion skills in v1... [CONFIRM] this stays true for the whole CVPR pass."* This was an open confirmation item, never closed. A3 is the first thing in this entire program to cross that line — worth flagging to whoever owns that confirmation before GPU time is spent on it. |

### A4 — Scalability/cost

| Plan item | Codebase reality |
|---|---|
| Signature caching, fingerprint pruning | **Exists**: `equivalence/candidates.py::compute_fingerprints` (perceptual average-hash ‖ mean DINO feature over a screening sub-battery) and `candidate_pairs()` (NN ∪ same-family ∪ engineered hard negatives). Confirmed by direct read: `candidate_pairs()` returns only the candidate set — **no reduction-ratio or recall tracking is computed today**. A4's reduction/recall curve ([`A4/01_measurement_protocol.md`](A4/01_measurement_protocol.md) §2.2) is genuinely new *instrumentation wrapped around an existing function*, not new candidate-generation logic. |
| Eager-vs-lazy feature computation ambiguity (flagged as a contradiction in my critique response) | **Resolved by reading the code**: `backends.py` batch-extracts features per stage and frees the model after (`close()`), and `compare.py`'s `BatteryEvaluator` **caches outputs per `(skill, params, seed)`**. This is an eager-per-stage, cached-thereafter design — not the ambiguous eager/lazy split I flagged abstractly. A4's complexity model in [`A4/02_analysis_and_deliverables.md`](A4/02_analysis_and_deliverables.md) §1.3 should be corrected to describe *this* design rather than presenting both eager and lazy as open possibilities. |
| Curation Pareto (F1/success vs. cost) | **Exists as a first-class deliverable, not something A4 invents**: `studies/metrics.py::pareto_front()` / `aggregate_pareto_front()` — non-dominated front over success, compression, action cost. This is **Study 2** in the codebase's own numbering. A4's Pareto plot (F1 vs. USD) should be built as a variant axis on this existing function (substitute action-cost/USD for the existing cost axis) rather than a new plotting pipeline. |
| Construct-validity study | **Exists**: `studies/metrics.py::construct_validity()` — Pearson/Spearman between `intrinsic_curation_score()` and downstream success. Not in my original A4 plan at all — a genuine gap in what I proposed. Worth adding as a cheap, already-built deliverable. |
| Vision-matters ablation (output-gated vs. text-gated) | **Exists**: `studies/metrics.py::vision_matters_ablation()` — matches rows by `(ρ, composition, seed, mode)`, reports success/compression/action-cost deltas. This is effectively **A2's H1 hypothesis test, already implemented as a first-class study** — I designed a hypothesis-testing protocol in `A2/01` without knowing this function existed to run it. |
| Token/USD-per-repair cost tracking | **Genuinely absent** — the agent adapters (`OllamaClient`, `AnthropicClient`) don't appear to track token counts or cost in what I've read. This is real new instrumentation work, matching what A4 already planned. |
| Curator model roster | **The Anthropic path already exists**: `curation/agent.py`'s `LlmCurationAgent` behind `AnthropicClient` (currently referencing "Claude Opus 4.8 + adaptive thinking" per the phase summary — an older model string that needs updating to current model IDs). **Adding Claude Opus 5 / Sonnet 5 / Haiku 4.5 is a config change to an existing client, not new integration work.** The Ollama path (`list_ollama_models`) already supports arbitrary local models too. |
| Bootstrap CIs (vs. normal-approximation) | **Explicitly a documented extension point**: *"Confidence intervals are normal-approximation 95% CIs... If the final paper uses bootstrap CIs, this module can add a bootstrap option without changing the row schema."* Matches my critique-response demand for instance-level bootstrap exactly — scoped, small, anticipated. |
| GPU memory at scale | **Genuinely unmeasured anywhere** — confirmed absent. The 6GB budget is "met by design" but never measured even at pilot scale. A4's memory analysis is real new work. |

---

## 4. The single most important practical consequence

**There is already a complete, tested, end-to-end CLI pipeline that has simply never been run on real backends:**

```bash
pip install -e ".[ml,viz]"
viscurate build-probes  -c configs/probes.yaml     -o data/probe_images
viscurate freeze-oracle ...
viscurate run-benchmark --device cuda --clip --calibrate --date <today> -o results/phase4_benchmark
viscurate corrupt       -c configs/corruption.yaml
viscurate curate        --instance <L_rho bundle> --anthropic ...   # per corrupted instance
viscurate phase8        --points results/.../points.json -o results/phase8_studies
viscurate phase9        -c configs/phase9.yaml -o results/phase9
```

**This is not a script I need to draft for your friend.** It already exists, is tested (264 passing tests, `ruff`/`mypy --strict` clean), and its own status report names it as the literal next action. What your friend's GPUs are actually needed for, in priority order:

1. **The Phase-4 divergence run** — `run-benchmark --device cuda --clip --calibrate` — the project's own designated go/no-go, currently only run in a CPU/no-ml smoke-test mode. This produces the paper's Table 2/3 for real, resolving §1 above.
2. **The corruption grid + curation episodes** (Phase 5/6/8) at full scale — this produces Table 1's real numbers, once real curator models are wired in.
3. **A1's real-corpus work** — new code, GPU needed for backend inference across possibly thousands of real skills.
4. **A3's neural extension** — new code, the most GPU-hungry item (model zoo + distributional sampling).
5. **A2's new baselines and A4's instrumentation** — mostly CPU-light, GPU only for the code/sentence-embedding baselines.

---

## 5. Revised priority order

The `Experiments/README.md` execution order (merge-policy blocker → A4 telemetry → A2 core → A1 → A4 measured → A3 feasibility) still holds, but two items move to the front given what's now known:

```
0.  Resolve the merge-policy paper/appendix wording (unchanged — trivial, confirmed safe by gating.py)
0.5 Locate or accept the absence of the codebase version that produced the paper's current
    Tables 1-3 numbers (§1) — this determines whether "re-run" or "run for the first time"
    is the honest framing in the paper
1.  Run the existing Phase-4 pipeline for real on GPU (the codebase's own go/no-go,
    unrelated to any new Experiments/ work) — resolves §1 for good
2.  Begin hardened-sandbox implementation (§2) in parallel — long-lead item, gates A1 Corpus B
    and A2's trojan defect specifically, nothing else
3.  A4 telemetry layer, built as instrumentation around existing functions (§3, A4 row 1-2)
    rather than new pipelines
4.  A2 new baselines (rungs 3/4/5/7) + G_ρ merge-recall re-analysis (config change to
    existing runner)
5.  A1 Corpus A (no sandbox dependency — pip-installable classical libraries only)
6.  A1 Corpus B once the sandbox lands
7.  A3, using the existing backend/comparator plumbing but building the distributional
    inference layer from scratch
```

---

## 6. What I'm NOT changing

The **research design** in `A1/`–`A4/` — hypotheses, statistical corrections from the GPT-critique pass, metric definitions, annotation protocols — remains valid and is not superseded by any of this. Nothing in the codebase contradicts the methodology; it either already implements pieces of it (good — less to build) or hasn't gotten there yet (expected — the codebase's own status report says the same). This document adds pointers to real code and flags two new prerequisites (§1, §2); it does not revise the underlying experimental logic.

---

## 7. The newer repo — resolves one question, opens a bigger one

**Source:** https://github.com/farhanishraq17/VisCurate- , `main` @ `c4b6a94` (pushed 2026-07-05)
**Checked:** 2026-08-19

### 7.1 The 944-vs-24 pair discrepancy is resolved — it was a misreading, not a real gap

§1 above treated "`G_0` is 24 pairs in `configs/ground_truth_g0.yaml`" and "the paper claims 944 designed pairs" as a contradiction. It isn't. **944 is the size of the *candidate set*** `equivalence/candidates.py::candidate_pairs()` produces for 100 skills (fingerprint-NN ∪ same-family ∪ engineered hard negatives), scored against the 24-pair answer key with everything else defaulting to `DISTINCT`. Arithmetic confirms it exactly: `7 subsumption + 10 semantic + 1 complementary + 926 distinct = 944`. This part of §1 is retracted — it was my error, not the project's.

### 7.2 The real problem: two documents in the same push contradict each other

`phase4.md` (the same commit) is a polished, paper-ready writeup. It reports, as completed fact:
- backends `lpips-alex` / `vit_base_patch16_224.dino` / `clip-ViT-B-32-quickgelu (openai)`, **"computed on GPU"**
- thresholds **calibrated** on a 713-pair cluster-disjoint split, provenance-stamped
- the exact numbers now sitting in the paper's Table 2/3: `0/926`, `0/6`, name-match `44/926` and `5/6` (83.3%), TF-IDF `1/926`, LLM-judge `1/926` and `1/6`, `SEMANTIC_PRESERVING` precision `0.037`, `SUBSUMPTION` F1 `0.750`, `DISTINCT` precision `0.988` / recall `0.605` — **every one matches the paper draft to three decimal places.**

`WORK_PROCESS_OVERVIEW.md`, committed in the **same push**, says the opposite in its own words:

> *"The completed Phase 4 artifacts did not use the real LPIPS/DINO/CLIP visual comparison stack... Phase 4 should be described as implemented/provisional machinery, not as a completed full visual equivalence benchmark."*

and gives the actual manifest of the only visible completed run (`results/phase4_vllm_qwen3_4b/`):

```
n_skills: 100        n_pairs: 24          seed: 1234
perceptual_backend: null   semantic_backend: null   clip_backend: null
thresholds_calibrated: false          battery_n: 43          device: cpu
```

— 24 pairs, not 944; no backends, not GPU-computed LPIPS/DINO/CLIP; uncalibrated, not calibrated. It further states a GPU run was *started* ("began candidate generation and pair scoring") but produced no completed report, and gives an explicit **"the paper should not claim"** list that includes almost every number `phase4.md` reports as fact.

**This is not a subtle inconsistency — one document in the repository states as its own explicit purpose that the numbers the other document presents must not be claimed.**

### 7.3 The sandbox commit is also not what its message claims

A commit dated 2026-07-02 ("Add hardened sandbox for agent-authored skills, enable split action...") reads as if §2's blocker (no hardened sandbox) is resolved. **Checked directly against `curation/sandbox.py` on `main` right now — it is not.** `ExecutionPolicy.allow_untrusted` still defaults to `False`, the module docstring still states *"this module does not implement or enable untrusted execution,"* and `WORK_PROCESS_OVERVIEW.md` §8/§10.6 independently confirms: *"The hardened sandbox was not implemented or used."* **§2's blocker (A1 Corpus B, A2 trojan) still holds** — the commit message overstated the change, or it didn't fully land on `main`. Worth asking about, low-stakes either way since the code and the honest status doc agree.

### 7.4 Table 1 (the curation-agent grid) is only two-thirds confirmed

`WORK_PROCESS_OVERVIEW.md` §9 lists six **confirmed, real, 300-instance** vLLM runs, and their P/R/F1/intrinsic-score/action-cost numbers match the paper's Table 1 exactly for those six models: Qwen 3.5 2B, Qwen 3.5 9B, Llama 3.2 3B, Llama 3.1 8B, Gemma 4 12B, Qwen 3.5 27B.

**Missing from the confirmed list:** Llama 3.2 1B, Qwen 3.5 4B, and — **the paper's headline model** — **GPT-5.5**, the one the abstract and every framing sentence leans on ("the strongest curator recovers only about half..."). The commit message mentions "operational sweep scripts for OpenAI/vLLM curation runs," so a GPT-5.5 run may exist somewhere not reflected in this particular status doc's table. **This needs a direct check, not an assumption in either direction** — it's the single number the paper's narrative depends on most.

### 7.5 What this means, plainly

The question posed in §1 — "is a later run backing these numbers, or were they never real" — has a documented, first-party answer, and it's the second one, at least for Phase 4: **the repo's own honest-accounting document says the real-backend run was not completed**, while a separate document in the identical commit presents fully-formed numbers as if it had been. The paper draft's Table 2/3 are currently indistinguishable from the second document.

**This is not a "run it and see" problem anymore — it's a "figure out which of these two documents is true" problem, and only someone with access to `results/` on the machine that produced `phase4.md` can resolve it**, by checking whether a `results/phase4_benchmark/` (or similarly named) directory exists with a manifest showing real, non-null LPIPS/DINO/CLIP backends and `calibrated: true`. If it exists, `phase4.md` is real and `WORK_PROCESS_OVERVIEW.md` is simply stale/wrong. If it doesn't, `phase4.md`'s numbers — which are the paper's numbers — were written before the run that was supposed to produce them, and Stage 1 of `RUNBOOK.md` is not a confirmation run, it's the **first real one**.

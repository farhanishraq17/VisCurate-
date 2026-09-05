# VisCurate — WACV 2027 Round-2 Experiment Suite

Working documents for the four revisions required by the Area Chair ([`../Viscurarte_Rebuttal/Meta_Reviewer.md`](../Viscurarte_Rebuttal/Meta_Reviewer.md)).

| Folder | AC item | Answers | Status | Effort |
|---|---|---|---|---|
| [`A1/`](A1/) | Evaluation on ≥1 real, uncurated open-source skill library | AC #1 · R1 W2 · R2 W2 · R3 W1 | Planned | High |
| [`A2/`](A2/) | Text/embedding baseline + verifier self-validation | AC #2 · R3 W2, W3, minor | Partly done in draft | Medium |
| [`A3/`](A3/) | Distributional / soft-similarity support for neural & stochastic tools | AC #3 · R1 W1 | Planned — **scoped as feasibility** | High |
| [`A4/`](A4/) | Scalability / cost analysis | AC #4 · R1 W3 | Planned | Medium |

> **Revised 2026-08-18 (two passes).** First pass: adjudicated an external critique ([`GPT_OPINION.md`](GPT_OPINION.md) → [`CRITIQUE_RESPONSE.md`](CRITIQUE_RESPONSE.md)) — 28 points accepted, 4 modified, 2 rejected. Second pass: reconciled against the **actual VisCurate codebase** ([`CODEBASE_RECONCILIATION.md`](CODEBASE_RECONCILIATION.md)), which exists at `Viscurate/Serious Deadlines/VisCurate_Experiments/` and was not previously known to this planning folder. **Read the reconciliation doc before drafting any GPU work** — it changes what's genuinely new vs. already built, and surfaces two findings that outrank everything else here:
>
> - ★ **The paper's current Tables 1–3 numbers do not trace to any run in this codebase.** The codebase's own status report says zero results exist yet; `G_0` there has 24 pairs, not the paper's claimed 944. This needs resolving before anything else — see `CODEBASE_RECONCILIATION.md` §1.
> - ★ **The hardened sandbox needed to safely execute untrusted third-party code does not exist yet.** It blocks A1 Corpus B and A2's trojan defect specifically — see §2 there.
>
> Other changes from the first pass: a manuscript correctness bug in the merge policy (confirmed safe by reading the actual gate code — see blocker below), A3's distributional design was mathematically invalid (pooled MMD) and has been rewritten, and Corpus B is mandatory, not cuttable.

**Strategic priority (by value): A1 > A2 > A4 > A3.** A1 is the only weakness all three reviewers raised independently, and R3's "Resubmission: No" hinges on A1 + A2.

**Execution order (by dependency and cost) — not the same thing:**

```
0.  Resolve the merge-policy blocker (below)          ← gates every safety metric
1.  A4 telemetry layer                                ← must exist before any run
2.  A2 core: G_ρ replay + fair baseline curves        ← nearly free, biggest gap closed
3.  A1 Corpus A + reduced Corpus B                    ← the AC's item #1
4.  A4 measured cost at real n (extrapolate after)
5.  A3 as a scoped feasibility study                  ← cut first if time runs out
```

A2 runs before A1 **because it is cheap, not because it matters more.** A1 Corpus B must not be starved to fund optional A2 breadth — it is the only corpus that literally satisfies the AC's "uncurated open-source skill library."

> **Mandatory core:** steps 0–4. Everything beyond is labelled optional in the individual files. See [`CRITIQUE_RESPONSE.md`](CRITIQUE_RESPONSE.md) for the full scope adjudication.

---

## → To actually run this: [`RUNBOOK.md`](RUNBOOK.md)

Operational instructions for the GPU machine — environment setup, exact commands in order, go/no-go checks, what to send back. Drop-in code is in [`code/`](code/) with an install guide at [`code/README.md`](code/README.md).

The five stages, in dependency order:

| Stage | What | New code? | GPU? |
|---|---|---|---|
| 0 | Setup + **telemetry install** ← do first | yes — [`code/`](code/) | no |
| 1 | ★ **Phase-4 divergence run — the go/no-go** | **none** (already exists in the repo) | yes |
| 2 | Corruption grid + curation episodes | none | yes + LLM API |
| 3 | A2 new baselines (rungs 3/4/5/7) | yes — [`code/`](code/) | light |
| 4 | A1 Corpus A cross-library audit | yes — to write | no |
| 5 | Blocked: A1 Corpus B + trojan (need hardened sandbox); A3 | — | — |

**Stage 1 needs no new code at all** — the pipeline exists and has simply never been run on real backends.

---

## Where the un-numbered reviewer items live

The AC listed several weaknesses without numbering them as revisions. They are folded into the four folders rather than given their own, so nothing is lost:

| Item | Source | Home |
|---|---|---|
| OOD divergence — same in-distribution, different out-of-distribution | R2 | [`A2/04_ood_and_scope_limits.md`](A2/04_ood_and_scope_limits.md) |
| Security / side-effects — identical outputs, harmful payload | R2 | [`A2/04_ood_and_scope_limits.md`](A2/04_ood_and_scope_limits.md) |
| Probe-battery ablation | R3 (minor) | [`A2/03_ablations.md`](A2/03_ablations.md) — feeds A4 |
| Token/compute cost of equivalent skills | R2 | [`A4/`](A4/) |
| Maintainability of equivalent skills | R2 | [`A4/02_analysis_and_deliverables.md`](A4/02_analysis_and_deliverables.md) |
| Expanded curator model roster | (pre-emptive) | [`A4/01_measurement_protocol.md`](A4/01_measurement_protocol.md) §4 |

---

## Dependency graph

```
A2b (G_ρ re-analysis) ──── free, no new compute ──── DO FIRST (week 0)
        │
        ├──> A2c (fix broken taxonomy rows)
        │
A1 Corpus A (cross-library) ── independent ──> feeds A2's baseline ladder with real pairs
        │
A1 Corpus B (ComfyUI) ── long pole ──> feeds A4 with real-scale timing
        │
A2 E5 (probe ablation) ────────────────────> feeds A4 (smaller battery = lower cost)
        │
A3 (neural/stochastic) ── independent, parallelizable with A1
        │
A4 ── instruments everything above; needs A1/A3 runs to exist before final numbers
```

**Parallelism:** A1 and A3 are the two heavy items and are fully independent — split across people. A2b/A2c/E5 are cheap and should be done first to fill tables early.

---

## Shared infrastructure (build once, used by all four)

| Component | Used by | Notes |
|---|---|---|
| Hardened execution sandbox | A1, A2 (trojan), A3 | Already exists per `sec/appendix.tex` §Output-Gated Curation. A1 extends its trust boundary from agent-authored code to **third-party** code. |
| Canonicalization map `φ` | A1, A3 | Must be extended for real-world dtype/layout/range chaos. See [`A1/02_corpus_B_comfyui.md`](A1/02_corpus_B_comfyui.md) §5. |
| Signature cache | A1, A3, A4 | Per-skill signatures computed once, reused across all pairs. Cache hit rate is a headline A4 number. |
| Instrumentation / telemetry layer | **A4**, wraps all others | Build in week 0 so every subsequent run is measured. Retrofitting timing is wasted work. |
| Frozen-artifact discipline | all | Every pair list, pack list, and calibration split gets hashed and committed **before** any metric runs. The paper already promises this (`sec/appendix.tex` §Threshold Calibration); real corpora make it easy to violate accidentally. |

> **Build A4's instrumentation first.** It is listed fourth by priority but its telemetry hooks must exist before A1 and A3 runs happen, or those runs produce no cost data and must be repeated.

---

## ★ BLOCKERS — resolve before running anything

### 1. Merge-policy wording — RESOLVED, needs only a text fix
`SEMANTIC-PRESERVING` was described four inconsistent ways in the manuscript. **Read directly from the real codebase** (`curation/gating.py` + `equivalence/relations.py::licenses_merge`): the code licenses merges only for `EXACT`/`PERCEPTUAL`, never `SEMANTIC_PRESERVING`. The code is safe. `sec/appendix.tex:52` ("licenses parameterize **or merge**") is the wrong sentence — fix the paper text to match `sec/7_results.tex:60`. See [`CRITIQUE_RESPONSE.md`](CRITIQUE_RESPONSE.md) §1.

### 2. Paper numbers don't trace to the codebase — UNRESOLVED, resolve first
The current draft's Tables 1–3 (944 pairs, `0/926` false merges, GPT-5.5 at μF1 0.583...) do not match the actual codebase, which has a 24-pair `G_0` and, by its own status report, **zero results anywhere**. Find out whether a later codebase version produced these numbers, or whether they need to be run for real. See [`CODEBASE_RECONCILIATION.md`](CODEBASE_RECONCILIATION.md) §1 — **read this before anything else in this folder.**

### 3. Hardened sandbox doesn't exist — blocks two specific items
Untrusted-code execution is deliberately disabled in the current codebase (`curation/sandbox.py`). This blocks **A1 Corpus B** and **A2's trojan defect type** specifically — nothing else in the suite depends on it. See [`CODEBASE_RECONCILIATION.md`](CODEBASE_RECONCILIATION.md) §2.

---

## Non-negotiable methodological rules

These apply to every experiment in this suite. Violating any one hands a round-3 reviewer an easy rejection.

1. **Never re-calibrate thresholds on real-corpus data**, and **no test-driven repair** anywhere. Any new pre-filter, margin, or threshold motivated by a bad result is selected on a **fresh validation split** and evaluated **once** on untouched families.
2. **Freeze and hash before you run.** Pair lists, pack lists with commit SHAs, parameter alignment maps, annotation guidelines — committed before the first metric.
3. **Report the denominator, and an exact interval.** Every rate carries raw counts and a **Clopper–Pearson** interval. Every exclusion gets its reason logged and tabulated.
4. **Compare all baselines at matched operating points** — including the two already in the draft, which are currently pinned at arbitrary thresholds. See [`A2/01_baseline_ladder.md`](A2/01_baseline_ladder.md) §4.
5. **Cluster uncertainty correctly.** Bootstrap by pack, skill, or corruption instance — **never by pair.** Pairs sharing a skill are dependent, and pair-level bootstrap badly understates variance.
6. **State hypotheses before results.** No design document may assert an expected outcome as its rationale. Pre-declare what would refute the hypothesis.
7. **Report negative results.** If VisCurate underperforms on real data, that is the finding. The reviewers already suspect a generalization gap; measuring it honestly earns more credit than hiding it.

### ⚠ Statistical power for the safety claim

The paper's designated headline (`sec/6_methodology.tex:192`) is the hard-negative false-merge rate, currently **0/6**. Its exact 95% upper bound is **39.3%** — the headline number is the statistically weakest one in the paper.

| Observed | Exact 95% upper bound |
|---|---|
| 0/6 | **39.3%** |
| 0/926 | 0.32% |
| **0/59** | **5.0%** ← minimum for a credible ≤5% claim |

**Expand the hard-negative slice to n ≥ 59.** Sources: additional engineered pairs in `L_0`, real cross-library near-misses from A1 Corpus A, and A3's depth-vs-disparity and instance-vs-semantic pairs. Until then, no sentence in the paper may assert a low hard-negative false-merge rate without its interval attached.

---

## Required artifacts per experiment

No result enters the paper without these. Adopted from [`GPT_OPINION.md`](GPT_OPINION.md) §1 — the one substantive gap in the original plans.

| Artifact | Notes |
|---|---|
| **Frozen input manifest** | pair lists, pack lists, model rosters — with hashes, provenance, and retrieval dates |
| **Locked environment** | per-pack containers for A1 Corpus B; pinned versions everywhere else |
| **Executable code** | scripts, not notebook fragments |
| **Raw per-unit outputs** | one record per pair / skill / episode, not just aggregates |
| **Failure and exclusion logs** | with reasons, tabulated |
| **Machine-readable metrics** | with denominators and intervals |
| **Run manifest** | commit SHA, hardware, seeds, timestamps, configuration |
| **Regeneration scripts** | every table and figure reproducible from raw outputs by one command |

---

## Draft blockers (fix regardless of experiments)

Tracked in [`../Viscurarte_Rebuttal/PLAN.md`](../Viscurarte_Rebuttal/PLAN.md) §7.

**Correctness:**
- ★ **The `SEMANTIC-PRESERVING` merge-policy contradiction** — see the blocker section above. Highest priority in the entire suite.

**Hard errors:**
- `\begin{itemsize}` in `sec/1_intro.tex:23` — **environment defined nowhere**; the contributions list is broken
- Intrinsic Score has **no definition** (`sec/7_results.tex:23–25` reads "defined as \\" then stops) despite being a headline column
- `sec/appendix.tex:11` Training Details is `\textcolor{red}{Add training details here}`, while `sec/7_results.tex:27` promises it to the reader
- `rebuttal.tex` is still the unmodified WACV template
- ~11 citations to 2026 works need verification — a fabricated reference in round 2 is unrecoverable

**Clarity (preempts a repeat misreading):**
- R1's critique refers to *"a battery of over 200 probe images."* The battery is **177 probes** (`sec/6_methodology.tex:160`); the 200 is the **action-step budget** (`sec/6_methodology.tex:151`). A reviewer conflated them. Separate the two prominently — different sections, explicit units — so it does not happen again in round 2.

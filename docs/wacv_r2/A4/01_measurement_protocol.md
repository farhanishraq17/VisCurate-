# A4 · Measurement Protocol

---

## 1. The telemetry layer (week 0)

Wrap every execution path with structured timing. Emit one record per event to a append-only log.

### 1.1 Events to record

| Event | Fields |
|---|---|
| `signature_compute` | skill_id, n_probes, n_grid_points, wall_ms, gpu_ms, peak_mem, backend, **cache_hit** |
| `fingerprint_compute` | skill_id, wall_ms, dim |
| `candidate_gen` | n_skills, radius, n_candidates, wall_ms |
| `pair_verify` | pair_id, **stage** (`exact`/`perceptual`/`subsumption`/`semantic`/`complementary`), wall_ms, gpu_ms, verdict, **short_circuited** |
| `llm_call` | model, tokens_in, tokens_out, **thinking/effort level**, wall_ms, retry_count |
| `agent_episode` | model, instance_id, n_actions, n_applied, n_rejected, n_blocked, n_invalid, total_tokens_in, total_tokens_out, wall_ms |
| `skill_execute` | skill_id, probe_id, wall_ms, peak_mem (for the per-skill cost tie-breaker, §4 of `02_`) |

### 1.2 Two fields that carry the whole argument

**`cache_hit` on `signature_compute`.** This is what proves signature computation is `O(n)` and not `O(n²)`. Without it the cache-reuse claim is an assertion.

**`short_circuited` on `pair_verify`.** The classifier is *"hierarchical and stop-at-first"* (`sec/6_methodology.tex:72`) — cheap strict stages run before expensive learned ones. Recording which stage decided each pair shows how often the expensive backends are even reached. If most pairs resolve at the cheap `L∞` or shape-gate stage, the average cost is far below worst case, and that is a strong and currently-unmade argument.

### 1.3 Discipline
- Emit records for **every** run, including A1 and A3. Do not gate telemetry behind a flag someone forgets to set.
- Timestamp and tag with a run ID + git SHA so records are attributable.
- Record hardware once per run: GPU model, CPU, RAM, CUDA version.

### 1.4 ⚠ GPU timing needs a real benchmark protocol

> **Added 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §32. Hardware metadata alone is not enough — naive `time.time()` around an async CUDA call measures kernel *launch*, not execution.

Specify and follow:

| Requirement | Why |
|---|---|
| **Warm-up iterations** discarded | first call includes lazy init, autotuning, JIT |
| **`torch.cuda.synchronize()` or CUDA events** | CUDA is asynchronous; without sync you time the launch, not the work |
| **Repeated trials** (≥ 10) | single measurements are noise |
| **Report median *and* tail (p95)** | tail latency drives real-world experience, especially on real corpora |
| **Confidence intervals** | a point estimate is not a measurement |
| **Cold vs. warm cache stated separately** | first-touch feature loading differs greatly from steady state |
| **I/O inclusion declared** | is probe loading counted? state it either way |
| **Concurrency and power mode fixed** | shared-GPU contention and clock throttling silently change results |
| **Peak memory via `torch.cuda.max_memory_allocated()`** | not inferred from model size |

Record the protocol in the appendix. Timing numbers without it are not reproducible, and this is exactly the kind of detail a systems-literate reviewer checks.

---

## 2. Verification-cost measurements

### 2.1 Per-component timing
On the existing 100-skill library, with the 177-probe battery:

| Quantity | How |
|---|---|
| Per-skill signature time | mean ± std over 100 skills, CPU and GPU broken out |
| Per-pair verification time **by cascade stage** | shows where time actually goes — expect LPIPS/DINO forward passes to dominate |
| Fraction of pairs short-circuiting at each stage | the cheap-stages-do-most-of-the-work argument |
| Cache hit rate | signatures reused across pairs |
| Peak GPU memory | the practical deployment constraint |

### 2.2 Candidate reduction/recall sweep ★ the required curve

Sweep the fingerprint radius `r` and, at each value, record **both**:

| Metric | Definition |
|---|---|
| **Reduction ratio** | `\|C(r)\| / C(n,2)` — how much work is skipped |
| **Candidate recall** | fraction of truly-related pairs (from `G_0` ∪ `G_ρ`) retained in `C(r)` |

**Plot them against each other.** A reduction number alone is not a result — pruning everything achieves perfect reduction and destroys the method. The tradeoff curve is what answers R1.

Mark the operating radius currently used, and report its recall.

> ⚠ **Recall measured on `G_0`/`G_ρ` is partly guaranteed by construction** ([`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §13). `sec/appendix.tex:43` states that candidate generation *always* includes same-family pairs and the engineered hard negatives. Since the synthetic positives largely **are** same-family or engineered, high recall on those graphs is partly an artifact of the inclusion rule rather than evidence the fingerprint works.
>
> **Report recall on three sources separately:**
> 1. synthetic positives from `G_0`/`G_ρ` — note the inclusion-rule caveat explicitly
> 2. **real adjudicated positives from A1 Corpus A and Corpus B** — the number that actually supports the scalability claim
> 3. **injected known positives** planted specifically to test pruning (see [`../A1/03_annotation_and_metrics.md`](../A1/03_annotation_and_metrics.md))
>
> Report each with an interval, broken down by family. If recall is 1.0 at the operating radius on *real* positives, that is the strongest possible answer — but it must be measured there, not inferred from the synthetic graphs.

### 2.3 End-to-end
Full-library verification wall clock and GPU-hours for `n = 100`, decomposed:

```
total = n·t_sig + n·t_fingerprint + t_candgen + |C|·t_pair
```

Report each term's share. This decomposition **is** the scalability argument: the `O(n²)` term is `|C|·t_pair`, and `|C|` is controlled by pruning.

### 2.4 Real-corpus timing (free, from A1)
Corpus B gives real timing at real scale — nodes with genuine runtime variation, unlike the hand-authored `L_0`. Report signature time distribution across real nodes; the tail matters more than the mean for a practitioner.

---

## 3. Curation-cost measurements

### 3.1 Per-episode
From `agent_episode` records across the 300-instance grid, per model:

| Metric | Definition |
|---|---|
| Mean tokens in / out per episode | raw usage |
| **USD per episode** | tokens × published rates |
| **USD per successful repair** | **Σ cost ÷ Σ correct repairs** — see the warning below |
| Wall-clock per episode | latency |
| Cost per applied action | efficiency |

**USD per successful repair is the number practitioners actually need** and no one reports it. It also reframes the results table: a model with mediocre F1 but very low cost may be the rational choice, and the paper can say so.

> ⚠ **Compute it from totals, not as a mean of per-episode ratios** ([`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §32). The mean of ratios is not the ratio of means, and they diverge substantially when repair counts vary across instances.
>
> - Use `Σ cost / Σ correct_repairs` over all episodes, with a bootstrap interval clustered **by instance**.
> - **Define the zero-repair case explicitly.** Llama 3.2 1B scores 0.000 on every metric, so its cost-per-repair is undefined. Report it as *"no successful repairs (cost \$X incurred)"* — never as `∞`, `NaN`, or a silently omitted row.
> - **Timestamp provider prices**, and report **tokens and latency separately from dollars** so the results stay reproducible after rates change.

### 3.2 Connect to the existing efficiency story
`sec/7_results.tex:131` already observes that GPT-5.5 attains the best scores *"while attempting the fewest actions per task (AC = 75.15)"* while Llama variants *"exhaust the full action budget (AC = 200.00) yet predict only 1–4 repairs."*

**That is a cost story told without cost units.** Converting action counts to tokens and USD completes an argument the paper already started — cheap to do and it strengthens existing text rather than adding a disconnected section.

---

## 4. Curator roster expansion

The current lineup has **one** closed model (GPT-5.5). For a benchmark claiming to measure frontier capability, one proprietary data point is thin, and it also makes the Pareto plot nearly empty.

### 4.1 Recommended additions

| Model | Model ID | Context | Input $/1M | Output $/1M |
|---|---|---|---|---|
| Claude Opus 5 | `claude-opus-5` | 1M | $5.00 | $25.00 |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | $3.00 | $15.00 |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | $1.00 | $5.00 |

Plus one Gemini-family and one DeepSeek-family model for breadth.

> **Look up exact model ID strings and current rates at implementation time.** Do not transcribe third-party pricing from memory into a paper; rates change and a wrong number is an easy, embarrassing catch.

### 4.2 Implementation notes
- **Thinking/effort is a cost knob, not a detail.** For models supporting adaptive or budgeted reasoning, record the effort level in `llm_call` and report it in the appendix hyperparameter table. Two runs of the same model at different effort are different cost/quality points and belong as separate rows or a labelled sweep.
- **Hold the environment fixed.** `sec/6_methodology.tex:45` commits to the agent being the sole axis of comparison — environment, corruptions, and gate stay frozen.
- **Keep the judge disjoint.** `sec/6_methodology.tex:163` promises the `llm-on-descriptions` judge is *"deliberately distinct from any model used as a curation subject."* Adding models must not break that — check before choosing, and the same applies to A2's new LLM-on-source baseline.
- **Budget before launching.** Pilot on ~10 instances, read the actual token counts, extrapolate to 300 × N models, and confirm the total before committing. If it is too expensive, subsample the grid for the added models and **state the subsampling in the paper.**

### 4.3 Fill the appendix placeholder
`sec/appendix.tex:11` is currently `\textcolor{red}{Add training details here}` while `sec/7_results.tex:27` promises the reader it exists. The roster expansion is the natural moment to write it: decoding parameters, temperature, effort levels, context limits, retry policy, prompt templates, seeds, and per-model API versions.

---

## 5. Neural verification cost (from A3)

Neural operators are far more expensive than classical ones. Report separately — folding them into one average would misrepresent both:

| Metric | Note |
|---|---|
| Forward passes per signature | `probes × grid × K` |
| GPU-hours per operator | by modality |
| Cost multiplier vs. classical | the honest statement of what A3 costs |
| Cost vs. `K` | ties to A3's `K`-sweep knee |

---

## 6. Cost levers to report

Three concrete reductions, each measured elsewhere in the suite:

| Lever | Source | Expected saving |
|---|---|---|
| Fingerprint pruning | §2.2 | `C(n,2) → \|C\|` |
| Smaller probe battery | [`../A2/03_ablations.md`](../A2/03_ablations.md) §1 | if the knee is 64, ~2.8× |
| Cheaper backend configuration | [`../A2/03_ablations.md`](../A2/03_ablations.md) §2 | measured per config |
| Lower `K` for neural | [`../A3/02_method_distributional_equivalence.md`](../A3/02_method_distributional_equivalence.md) §6 | knee-dependent |

Presenting these as a menu of measured tradeoffs — rather than a single "it's fast enough" claim — is what actually answers R1.

# VisCurate - Methodology Notes and Implementation Scope

This document is a detailed source note for writing the methodology section of the paper. It is
not written as the final paper section. Its purpose is to record what the method was designed to
do, what was implemented, what was actually run, and what was not implemented or not run.

The important correction is:

- The completed experimental work stopped at the Phase 6 curation-agent experiments.
- The later downstream study phases were not run.
- The completed Phase 4 artifacts did not use the real LPIPS/DINO/CLIP visual comparison stack.
- Therefore, Phase 4 should be described as implemented/provisional machinery, not as a completed
  full visual equivalence benchmark.

---

## 1. Methodological Goal

VisCurate studies curation of an image-transformation skill library. A skill is treated as an
executable, deterministic image-to-image transformation:

```text
skill(image, params, seed) -> image
```

The main methodological idea is to curate a library based on **executed behavior** rather than
textual descriptions. Instead of deciding whether two skills are equivalent from their names or
documentation, the system executes both skills on a controlled probe battery and compares their
outputs. The output-based comparison is then used as a gate for structural curation actions such
as merging duplicate skills or parameterizing a specialization.

The method separates two roles:

- The **output verifier** decides what relation holds between two skills based on outputs.
- The **curation agent** proposes repository edits, but structural edits must pass the verifier.

This split is important for the paper because it prevents the curation agent from silently
collapsing distinct skills based only on text similarity.

---

## 2. Overall Pipeline

The intended full pipeline was:

```text
Clean library L0
  -> probe battery P
  -> frozen clean oracle
  -> output-grounded relation verifier
  -> Phase 4 equivalence benchmark
  -> corrupted libraries L_rho
  -> curation agent
  -> downstream task/query evaluation
  -> study aggregation and paper artifacts
```

The actually completed and run pipeline was:

```text
Clean library L0
  -> probe battery P
  -> frozen clean oracle
  -> verifier and benchmark machinery
  -> provisional Phase 4 run without LPIPS/DINO/CLIP backends
  -> corruption generator
  -> Phase 6 curation environment
  -> local vLLM curation-agent sweeps
```

The downstream query/task evaluation, full study aggregation, and final paper-artifact generation
were not run.

---

## 3. Method Components

### 3.1 Skill Representation

Each skill is represented as a typed object containing:

- A stable skill id.
- A human-readable name.
- A text description.
- A callable image-transformation function.
- A typed parameter schema.
- Metadata such as family, provenance, trust status, and internal labels.

The parameter schema supports typed coercion and validation for integers, floats, booleans,
strings, and enums. Each parameter may have defaults, ranges, and allowed choices. This matters
methodologically because equivalence between two parameterized transformations depends on valid
parameter bindings, not only on a default setting.

The clean library is serializable through a registry. The registry stores skill specifications
without pickling arbitrary function objects; callable functions are rebound through a known
resolver. This makes experimental artifacts easier to reproduce and inspect.

### 3.2 Text-Blind Comparator View

The output-grounded verifier receives a reduced view of each skill that excludes the text
description. This is a methodological safeguard: the output track is not allowed to read the same
information used by text baselines.

This gives a clean comparison between:

- Text-based methods that use names/descriptions.
- Output-based methods that use executed behavior.

### 3.3 Skill Library `L0`

The clean base library `L0` contains 100 deterministic image-transformation skills. These are
grouped into four broad categories:

| Category | Count | Examples |
|---|---:|---|
| Geometric/canvas operations | 25 | flips, rotations, translation, shear, scale, crop, resize, pad, tile |
| Color/exposure/contrast operations | 25 | grayscale, invert, brightness, gamma, saturation, hue, threshold, posterize, histogram equalization |
| Blur/filter/edge/morphology operations | 25 | Gaussian blur, box blur, median blur, unsharp mask, Sobel, Canny, FFT filters, dilation, erosion, denoising, noise |
| Mask/reconstruction/synthesis operations | 25 | inpainting, flood fill, contours, distance transform, dithering, pixel sorting, mosaics, vignettes, grids, sketches, RGBA masks |

The clean library deliberately excludes injected bugs, duplicates, and dead skills. Those defects
are introduced later through the corruption generator so that the pollution rate and defect type
are controlled.

### 3.4 Determinism Assumptions

The method assumes deterministic execution for a fixed tuple:

```text
(input image, parameter binding, seed)
```

Most skills are deterministic directly. A small number are seeded-stochastic, such as random
crop or noise injection, and are deterministic at a fixed seed. Precision-sensitive operations
such as FFT-based filters are not necessarily byte-identical across all conditions and are meant
to be handled by perceptual tolerance rather than exact hashing.

### 3.5 Output Canonicalization

Skill outputs may differ in type, range, shape, and channel structure. The implementation uses a
versioned canonicalization contract before comparison:

- Numeric outputs are converted to a normalized float view for metrics.
- A uint8 view is retained for hashing.
- Grayscale outputs are promoted to three channels when needed.
- RGBA outputs are composited over a fixed background while alpha is tracked separately.
- Binary masks are detected and compared with mask-appropriate logic.
- Shape mismatch blocks exact/perceptual equivalence unless a later semantic relation is being
  considered.

This canonicalization step is necessary because otherwise different output formats would make
relation labels ambiguous.

---

## 4. Probe Battery and Clean Oracle

The probe battery `P` is a controlled set of test images used to evaluate skill behavior. It was
designed to include synthetic probes and license-clean natural images.

The probe set included coverage for:

- Natural image-like content.
- Synthetic gradients and textures.
- Degenerate cases such as all-black, all-white, single-color, and small images.
- Multiple channel/format conditions such as RGB, grayscale, RGBA, 16-bit-like, and palettized
  cases.
- Inputs that exercise geometric, color, mask, and frequency-domain transformations.

The clean oracle was produced by executing all 100 clean skills over the probe battery and storing
reference outputs/hashes. The oracle is used to confirm that injected corruptions have the
intended behavioral effect. It is not used to fabricate relation labels.

Implemented:

- Probe generation machinery.
- Probe manifest and reproducibility metadata.
- Clean reference-output freeze.
- Deterministic verification that clean outputs can be regenerated.

Not implemented or not used as final paper evidence:

- A larger final-scale probe battery for the full paper study was not run.
- The parameter-sweep oracle was not used as a completed final artifact for all later studies.

---

## 5. Output-Grounded Relation Taxonomy

The verifier assigns relations between pairs of skills. The intended relation taxonomy is:

1. **EXACT**: outputs match exactly or within a small numeric epsilon across all relevant probes.
2. **PERCEPTUAL**: outputs are not exactly identical but are perceptually equivalent.
3. **SUBSUMPTION**: one skill is a special case of another under some parameter binding.
4. **SEMANTIC_PRESERVING**: two outputs differ pixel-wise but preserve a similar semantic effect.
5. **COMPLEMENTARY**: two skills are distinct but composable or orthogonal.
6. **DISTINCT**: no safe equivalence or subsumption relation was found.

The pipeline is conservative: exact and perceptual relations are intended to use worst-probe
aggregation because a single divergent probe can make a merge unsafe. Subsumption is directional
and requires searching for parameter bindings that reproduce the specialized behavior. Distinct
is the residual label and is important because it blocks unsafe merges.

Implemented:

- Exact/hash-based comparison machinery.
- Matched or searched comparison paths for parameterized skills.
- Subsumption search machinery.
- Relation objects with distances and rejection explanations.
- Output-based candidate generation and benchmark runner machinery.

Not completed as final experimental evidence:

- The real LPIPS perceptual comparator was not used in the completed Phase 4 result.
- The real DINO semantic comparator was not used in the completed Phase 4 result.
- The real CLIP comparator was not used in the completed Phase 4 result.
- Thresholds were not calibrated on human-labeled relation data.
- Human verification and kappa were not completed.

---

## 6. Phase 4 Equivalence Benchmark Scope

Phase 4 was intended to be the first major empirical result. The intended comparison was:

- Output-grounded verifier.
- Name-matching text baseline.
- Description embedding/cosine text baseline.
- LLM-on-descriptions text baseline.
- Divergence analysis between output-based and text-based decisions.

The intended full Phase 4 benchmark would use LPIPS/DINO/CLIP as the visual/perceptual backend
for perceptual and semantic relations.

### What Was Implemented

The following Phase 4 pieces exist:

- Designed relation graph `G0`.
- Candidate-pair benchmark runner.
- Text baselines.
- LLM-on-descriptions interface.
- Pair-level CSV output.
- Divergence CSV output.
- Markdown report writer.
- Review-template export for human verification.
- Manifest recording backend status and reproducibility metadata.

### What Was Actually Run

The completed Phase 4 artifact currently visible under `results/phase4_vllm_qwen3_4b/` is a
provisional run. Its manifest records:

```text
n_skills: 100
n_pairs: 24
seed: 1234
perceptual_backend: null
semantic_backend: null
clip_backend: null
thresholds_calibrated: false
battery_n: 43
device: cpu
```

The report shows that no LPIPS/DINO/CLIP backends were active:

```text
backends - perceptual: None, semantic: None, clip: None
thresholds calibrated: False
```

This means the Phase 4 result can be used only as a provisional wiring/text-baseline artifact. It
should not be written as the completed vision-grounded equivalence benchmark.

There is also a GPU benchmark log showing that a benchmark process began candidate generation and
pair scoring, but there is no completed report artifact from that log that records LPIPS/DINO/CLIP
results. Therefore, it should not be counted as a completed Phase 4 benchmark run.

### Provisional Phase 4 Observations

The provisional report shows useful diagnostic behavior, but not final paper evidence:

- The name-match baseline over-merged several hard negatives in the provisional report.
- The output verifier kept those hard negatives separate under the available non-ML machinery.
- The report exported review templates and divergence tables.

These observations can motivate the method, but they should be labeled as provisional because the
main visual/perceptual stack was not run.

### Not Implemented or Not Run for Phase 4

- Full LPIPS-based perceptual equivalence.
- Full DINO-based semantic comparison.
- Optional CLIP-backed semantic comparison.
- Calibration of epsilon/perceptual/semantic thresholds.
- Human verification of SEMANTIC/SUBSUMPTION/UNCERTAIN slices.
- Inter-annotator agreement.
- Final equivalence F1, confusion matrices, or divergence tables suitable for a paper result.

---

## 7. Corruption Generator

The corruption generator creates controlled variants of the clean skill library. A corrupted
library instance is indexed by:

```text
(rho, composition, seed, mode)
```

where `rho` is the fraction of library sites selected for corruption, `composition` controls the
mix of defect types, `seed` controls deterministic sampling, and `mode` controls whether defects
are single-type or mixed/co-occurring.

### Defect Types

The seven designed defect types are:

| Defect type | Mechanism | Why it matters |
|---|---|---|
| Implementation bug | Mutate the function output behavior | Tests whether output verification detects functional divergence |
| Metadata mislead | Change name/description/tags but keep function unchanged | Tests whether text methods are fooled while output methods are stable |
| Exact/perceptual duplicate | Add a duplicated or near-duplicated skill | Tests merge detection |
| Subsumption pair | Add a specialization/generalization relation | Tests directional parameterization |
| Parameter-schema bug | Corrupt parameter schema while keeping function mostly unchanged | Tests interface-level failures |
| Domain-scoped bug | Break behavior only on a subset such as RGBA/grayscale | Tests probe coverage |
| Dead skill | Add a correct but unused skill | Tests usage-based curation rather than output equivalence |

### Implementation Details

The corruption process is replayable. A corruption log records the selected sites, defect types,
and mutator names. The corrupted library, updated relation graph, QA report, and ideal action key
are derived from that log.

This avoids storing arbitrary generated Python functions in the artifact. Instead, named mutator
factories reconstruct the modified behavior.

Implemented:

- Deterministic corruption planning.
- Replayable corruption logs.
- Corrupted library generation.
- Updated relation graph `G_rho`.
- Ideal action keys for scoring curation actions.
- QA checks for defect activity.

Not completed as final paper evidence:

- The corruption grid was not followed by full downstream success evaluation.
- Pollution-vs-performance curves were not produced.

---

## 8. Curation Environment

The curation environment is the main Phase 6 system. It provides the state, action space, verifier
gate, and logging around an agent that proposes library edits.

### State

The agent observes a library summary, not hidden ground-truth labels. Internal labels such as
`is_buggy` and `is_dead` are omitted from the agent-facing view. This is important because the
agent must infer curation actions from observable metadata, relations, and usage signals rather
than reading labels.

### Action Space

The action space contains eight actions:

| Action | Meaning |
|---|---|
| `add` | Add a new skill |
| `remove` | Remove a skill |
| `modify` | Modify metadata/schema or, in principle, function behavior |
| `retrieve` | Retrieve/query a skill |
| `merge` | Merge redundant skills |
| `split` | Split one skill into multiple skills |
| `parameterize` | Fold a specialization into a more general parameterized skill |
| `end` | Stop the episode |

Structural edits such as `merge` and `parameterize` are gated by verifier results. The agent may
propose them, but the environment decides whether they are allowed.

### Verifier Gating

The environment uses a relation-to-action map:

- EXACT or PERCEPTUAL can allow a merge.
- SUBSUMPTION in the correct direction can allow parameterization.
- SEMANTIC_PRESERVING may allow a weaker unify/parameterize path.
- DISTINCT blocks merging.
- COMPLEMENTARY keeps both skills.

If an action is rejected, the environment records a structured rejection reason. This is useful
for later action-log scoring because rejected or blocked actions consume budget without repairing
the library.

### Trust Boundary

Agent-generated skills are not trusted by default. Untrusted skills are blocked from execution and
verification unless a hardened sandbox is enabled. The hardened sandbox was not implemented or
used. This matters for methodology because the experiments do not evaluate free-form code
generation by the agent; they evaluate curation actions over existing/replayed skill libraries.

Implemented:

- Curation state.
- Action representation.
- Verifier-gated action application.
- Action logs and episode summaries.
- Local LLM/vLLM agent adapters.
- Trust boundary for untrusted code.

Not implemented:

- Hardened sandbox for executing generated code.
- Full function-level modification by the agent.
- Practical `split` of generated new code into trusted executable skills.
- Full downstream usage-driven query loop as a completed experiment.

---

## 9. Phase 6 vLLM Curation-Agent Experiments

The completed experiments evaluated local vLLM-served language models as curation agents in the
Phase 6 environment. Each model was run over 300 corrupted-library instances or episodes, and the
agent's action logs were scored against the ideal action key derived from the corruption process.

These are **intrinsic curation-action experiments**. They measure whether the model proposes
correct repairs/actions relative to the known corruption log. They do not measure downstream task
success after curation.

### Experimental Unit

The experimental unit was a curation episode over a corrupted skill-library instance. For each
episode:

1. The corrupted library and its metadata were loaded.
2. The curation agent observed the agent-facing state.
3. The model proposed actions up to the configured step or cost budget.
4. The environment applied, rejected, or blocked actions using the verifier gate and trust rules.
5. The resulting action log was scored against the ideal action key.

### Metrics

The Phase 6 summaries report:

- **Mean precision**: fraction of applied/proposed repair actions that matched ideal actions.
- **Mean recall**: fraction of ideal actions recovered by the agent.
- **Mean F1**: harmonic mean of action precision and recall.
- **Mean intrinsic score**: curation-action quality after accounting for missed repairs and
  budget-spending invalid/rejected/blocked behavior.
- **Mean action cost**: average action budget consumed.

The exact scoring is intrinsic to the curation log and ideal action key. It is not a downstream
image-task success metric.

### Completed Model Runs

| Result directory | n | Mean precision | Mean recall | Mean F1 | Mean intrinsic score | Mean action cost |
|---|---:|---:|---:|---:|---:|---:|
| `phase6_vllm_qwen35_2b_full` | 300 | 0.134 | 0.008 | 0.013 | 0.007 | 128.35 |
| `phase6_vllm_qwen35_9b_full` | 300 | 0.357 | 0.074 | 0.101 | 0.060 | 171.52 |
| `phase6_vllm_llama3_2_3b_full` | 300 | 0.443 | 0.013 | 0.024 | 0.012 | 200.00 |
| `phase6_vllm_llama31_8b_instruct_full` | 300 | 0.615 | 0.061 | 0.095 | 0.049 | 200.00 |
| `phase6_vllm_gemma4_12b_full` | 300 | 0.777 | 0.204 | 0.290 | 0.159 | 172.23 |
| `phase6_vllm_qwen35_27b_full` | 300 | 0.662 | 0.423 | 0.505 | 0.358 | 109.59 |

The strongest run by mean F1 was `phase6_vllm_qwen35_27b_full`, with mean F1 `0.505` and mean
intrinsic score `0.358`.

### How to Describe These Results in the Paper

These results can support a methodology/results subsection about **intrinsic curation-agent
behavior** under controlled corruptions. They should not be presented as evidence that curation
improved downstream image-task performance, because the downstream evaluation was not run.

---

## 10. What Was Not Implemented or Not Run

This section should be reflected honestly in the methodology or limitations section.

### 10.1 Phase 4 Visual Equivalence Benchmark Not Completed

The final Phase 4 benchmark with LPIPS/DINO/CLIP was not completed. The existing Phase 4 report
has null visual backends and uncalibrated thresholds.

Do not claim:

- LPIPS was used for the reported Phase 4 results.
- DINO was used for the reported Phase 4 results.
- CLIP was used for the reported Phase 4 results.
- The output-grounded equivalence benchmark was fully validated.
- The visual/text divergence result was finalized.

Safe phrasing:

- "We implemented the benchmark machinery and ran a provisional non-ML/text-baseline check."
- "The full LPIPS/DINO/CLIP equivalence benchmark remains future work."

### 10.2 No Human Relation Annotation

Human verification of semantic or borderline relations was planned but not completed. Therefore:

- No inter-annotator agreement was computed.
- No calibrated semantic/perceptual thresholds were produced from human labels.
- SEMANTIC_PRESERVING labels should be treated as design intent or planned labels, not final
  human-verified ground truth.

### 10.3 No Downstream Query Evaluation

The downstream query/task layer was part of the planned pipeline but was not run as an experiment.
Therefore:

- No downstream success-vs-pollution curves exist.
- No before/after curation task-success comparison exists.
- No solver-agent task performance was reported.
- No query-derived usage study was completed.

### 10.4 No Phase 8 Study Aggregation

The planned paper studies were not completed:

- Equivalence F1 with calibrated visual backends.
- Curation Pareto front over success, compression, and action cost.
- Construct-validity correlation between intrinsic curation score and downstream success.
- Output-gated vs text-gated curation ablation.

The Phase 6 intrinsic metrics are not equivalent to these Phase 8 studies.

### 10.5 No Final Phase 9 Paper Artifact Bundle

Because the full Phase 4 and downstream studies were not run, the final reproducibility bundle and
paper-ready tables/figures were not produced from real study rows.

### 10.6 Hardened Sandbox Not Implemented

The system blocks untrusted generated code, but it does not implement a full hardened sandbox for
executing arbitrary agent-generated skill code. As a result:

- The experiments do not evaluate arbitrary code synthesis by the curation agent.
- Function-level modifications and generated split skills are not part of the completed method.

---

## 11. Suggested Methodology Claims

The paper can safely claim:

- A clean library of 100 deterministic image-transformation skills was implemented.
- Skills were represented with typed parameter schemas and deterministic execution contracts.
- A probe battery and clean oracle were built for behavioral comparison.
- An output-grounded verifier architecture was implemented with exact, subsumption, and
  visual-comparator extension points.
- A controlled corruption generator was implemented for seven defect types.
- A verifier-gated curation environment was implemented with a fixed action space.
- Local vLLM models were evaluated as curation agents over 300 instances per run.
- Intrinsic action-level curation metrics were computed from action logs and ideal action keys.

The paper should not claim:

- Completed LPIPS/DINO/CLIP equivalence benchmark results.
- Calibrated perceptual or semantic thresholds.
- Human-verified relation labels or kappa.
- Downstream task-success improvements.
- Full Phase 8 study results.
- Execution of arbitrary agent-generated skill code in a hardened sandbox.

---

## 12. Methodology Details Worth Preserving

These details are useful when drafting the Methods section:

- The verifier/agent split is the central design: the verifier supplies behavioral facts, while
  the agent proposes repository edits.
- The verifier is text-blind by construction, which separates it from text baselines.
- The clean library `L0` is defect-free; defects are injected later to make pollution controlled.
- Corruptions are replayed from logs, so experimental instances are deterministic and auditable.
- Metadata-mislead and dead-skill corruptions are intentionally not fully visible to output
  comparison alone. This motivates the need for an agent with metadata/usage reasoning.
- Structural edits are gated to avoid false merges.
- Phase 6 results are action-level, not downstream-performance-level.
- The unimplemented LPIPS/DINO/CLIP benchmark should be framed as planned/future full validation,
  not as a completed result.


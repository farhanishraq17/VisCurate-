# A1 · Corpus B — ComfyUI Custom-Node Ecosystem Audit

**Effort: HIGH (hard-cap 2 weeks). Signal: high.**
**Proves:** C2 — real libraries decay the way our seven defect types model.

This is what makes "uncurated at scale" literally true. It is also the long pole and the most likely thing to slip — hence the hard cap and the fallback to Corpus A.

---

## 1. Why ComfyUI fits

A ComfyUI node is close to a literal implementation of the paper's skill tuple `⟨k, μ, d, f_s, Θ_s⟩`:

| Paper | ComfyUI |
|---|---|
| `k` identifier | node class key in `NODE_CLASS_MAPPINGS` |
| `μ` name | `NODE_DISPLAY_NAME_MAPPINGS` entry, or the mapping key |
| `d` description | `DESCRIPTION` class attribute, or the class docstring |
| `Θ_s` typed schema | `INPUT_TYPES()` — typed, with defaults, `min`/`max`, and enum choices |
| `f_s` function | `getattr(instance, cls.FUNCTION)` |
| image type | `IMAGE` in `INPUT_TYPES` / `RETURN_TYPES` |

**Bonus:** ComfyUI's `IMAGE` is a uniform representation — `torch.float32`, `BHWC` layout, range `[0,1]`. Canonicalization is *easier* here than in Corpus A, where six libraries disagree about dtype, layout, and range.

### Ecosystem scale (verified 2026-08-15)
- **5,192 node packs** in the Comfy Registry (`api.comfy.org/nodes`, `total` field)
- WAS Node Suite alone ships **210+ nodes**, heavily image-processing: blur/filters, blend modes, color ops, transforms, morphology, effects
- Community-contributed, no central behavioral review, genuine decay (abandoned packs, renames, silent breakage after core API changes)

**No existing academic study analyzes duplication in this ecosystem** — worth one sentence in Related Work, and it means the novelty claim is safe.

---

## 2. Sampling frame — stratified, two estimands

> **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §10. The original popularity-only design conflated two different estimands.

Popularity-ranked sampling answers *"what redundancy affects common deployments?"* It does **not** estimate ecosystem-wide redundancy. Generalizing from top packs to all of ComfyUI is an inference the design cannot support.

**Use two strata and report them separately:**

| Stratum | Selection | Estimand |
|---|---|---|
| **Popular** | top-K packs by install count | redundancy affecting deployed agents |
| **Long-tail** | reproducible random sample from the remaining ~5,100 packs, drawn with a **recorded seed** | ecosystem-wide redundancy |

Report each stratum's redundancy rate with its own confidence interval. If a combined ecosystem estimate is reported, it must be **inverse-probability weighted** by the stratum sampling fractions, and labelled as such.

**Never write "X% of ComfyUI nodes are redundant" from the popular stratum alone.** Write "X% of nodes in the K most-installed packs" and "Y% in a random sample of the long tail."

### Procedure
1. Pull the pack list via the registry API (`https://api.comfy.org/nodes`, paginated).
2. Rank by install / download count; take top-K as the popular stratum.
3. Draw the long-tail stratum at random from the remainder with a recorded seed.
4. **Record commit SHAs for every pack.** These repos change weekly; reproducibility depends on pinning.
5. Freeze and hash the combined pack manifest before extraction.

The long-tail stratum can be smaller than the popular one — but it must be **non-empty and randomly drawn**, or the ecosystem claim cannot be made at all.

### Expected yield funnel
| Stage | Expected |
|---|---|
| Packs sampled | 50–100 |
| Total node classes | 1,500–4,000 |
| Declared image→image | 600–1,500 |
| Passing the self-contained filter (§3) | **400–1,000** |
| Successfully executing on the full battery | report honestly |

Report the actual funnel as table **T1**.

---

## 3. Extraction filter — state it precisely in the paper

Include a node **iff**:

1. `RETURN_TYPES` contains `IMAGE`, **and**
2. `INPUT_TYPES()["required"]` contains at least one `IMAGE`, **and**
3. every *other* required input is a primitive — `INT`, `FLOAT`, `STRING`, `BOOLEAN`, or a `COMBO` (list of choices)

### Why clause 3 is the important one
It excludes nodes requiring `MODEL`, `CLIP`, `VAE`, `LATENT`, or `CONDITIONING`. That yields a **self-contained, executable, deterministic image→image population with no model weights to download** — which:
- keeps Corpus B inside the paper's current deterministic scope
- removes the single largest source of dependency pain
- gives a clean, defensible, stateable inclusion criterion

**Nodes excluded by clause 3 are natural A3 candidates** (neural/stochastic operators). Note this in the paper as a bridge between the two experiments rather than presenting it as discarded work.

### Optional inputs
Optional inputs are left at their declared defaults. Record the defaults in the skill's schema — parameter-default drift is one of the paper's seven defect types, so real defaults are worth capturing.

---

## 4. Execution harness

### 4.1 Headless shim
Bypass the ComfyUI graph executor entirely — no server, no queue, no workflow JSON:

```python
# sketch
mod   = import_pack(pack_path)              # imports __init__.py
klass = mod.NODE_CLASS_MAPPINGS[node_key]
inst  = klass()
fn    = getattr(inst, klass.FUNCTION)
out   = fn(**kwargs)                        # kwargs built from INPUT_TYPES
image = out[klass.RETURN_TYPES.index("IMAGE")]
```

Key details:
- `INPUT_TYPES` is a `@classmethod` and may compute dropdown contents at call time — call it, don't introspect statically
- `RETURN_TYPES` is a tuple of `str`; find the `IMAGE` slot by index
- probe images must be converted to ComfyUI's `BHWC float32 [0,1]` convention before the call

### 4.2 Isolation and safety
- **Per-call hard timeout** with process kill — some nodes hang on degenerate inputs
- **Memory cap** per call
- **Existing hardened sandbox** from `sec/appendix.tex` §Output-Gated Curation

> **Paper-worthy point:** the sandbox's trust boundary now covers **third-party** code, not just agent-authored code. That is a strictly stronger claim about the sandbox than the current draft makes, and it should be stated.

### 4.3 Environment strategy — isolated per pack

> **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §9. This **reverses** the original recommendation.

The original plan proposed a single shared environment with conflicts logged as a finding. That is wrong for the primary estimate: a shared environment produces **order-dependent** dependency resolution, so exclusions become partly artifacts of *our harness* rather than properties of the ecosystem. Since the exclusion funnel is a headline result (§6), contaminating it with harness artifacts destroys its credibility.

| Purpose | Environment |
|---|---|
| **Primary behavioral estimate** | **Isolated, pinned environment per pack** (venv or container). Each pack resolves its own dependencies independently. |
| **Secondary "deployability stress test"** | Single shared environment, conflicts logged — reported separately and clearly labelled |

The secondary run is still worth doing: *"N of K packs cannot be co-installed"* is a genuine ecosystem-health finding. But it must not contaminate the behavioral numbers.

Record the resolved dependency set per pack in the artifact so each environment is reconstructible.

### 4.4 Determinism check
Every extracted node must produce identical output across two runs at fixed seed. Nodes failing this are either genuinely stochastic (→ A3 candidates) or non-deterministic by defect (→ a finding). **Classify and report both**, do not silently drop them.

---

## 5. Output contract violations — fail the gate, do not repair

> **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §8. The original prescribed clamping and shape-guessing. **Both are false-merge generators** — two nodes that genuinely differ in range handling look identical after clamping, and a wrong BCHW/BHWC guess can manufacture agreement between unrelated outputs.

**Default policy: a contract violation fails the compatibility gate.** The node is excluded from behavioral comparison and recorded in the exclusion funnel with its violation type.

| Case | Handling |
|---|---|
| Returns `uint8` despite declaring `IMAGE` | **fail gate**; log as type-contract violation |
| Returns `BCHW` instead of `BHWC` | **fail gate** — do **not** guess by shape heuristic |
| Batch size ≠ input batch | **fail gate**; shape mismatch |
| Values outside `[0,1]` | **fail gate** — do **not** clamp |
| Returns RGBA given RGB | channel-layout canonicalization is **in-contract** and permitted, per `sec/appendix.tex` §Behavioral Signatures |

### If a repair is applied anyway
Some violations are so common that excluding them would gut the corpus. If a repair is adopted for a violation class, it must be:

1. **explicit** — named in the paper, not buried in code
2. **deterministic** — no heuristics that could resolve differently per node
3. **logged** per node
4. **subjected to sensitivity analysis** — report the redundancy census both with and without the repaired nodes, and show the conclusion does not depend on the repair

**Type-contract violations are themselves a finding** about real library decay and get their own row in T1. A node that declares `IMAGE` and returns something else is exactly the kind of silent defect the paper is about — so counting them is more valuable than repairing them.

---

## 6. The exclusion table is a result, not an embarrassment

Log every node that fails, with its reason:

| Exclusion reason | Count | % |
|---|---|---|
| Pack import error | | |
| Missing model weight | | |
| Unmet Python dependency | | |
| Inter-pack version conflict | | |
| Timeout on probe battery | | |
| Crash on degenerate probe | | |
| Non-deterministic at fixed seed | | |
| Declared `IMAGE` but returned non-image | | |
| **Total excluded** | | |

**Report this prominently.** A reviewer who sees an honest 30% exclusion rate *with reasons* trusts the remaining 70% far more than a paper that quietly reports only what worked. The exclusion rate itself measures real library decay — which is AC item #1 expressed as a single number.

---

## 7. Redundancy census — the headline analysis

Once the skill population is extracted and executable:

1. Compute per-skill signatures on the 177-probe battery (cache them — A4 needs the timing)
2. Fingerprint + generate candidates per `sec/appendix.tex` §Candidate Generation
3. Run the full hierarchical verifier at the **frozen synthetic operating point** — no re-calibration
4. Report:

| Metric | Definition |
|---|---|
| **Redundancy rate** | % of nodes certified `EXACT`/`PERCEPTUAL` with ≥1 other node — **per stratum**, with CI |
| **Cross-pack redundancy** | % of redundant pairs spanning *different* packs (the interesting case — independent reimplementation) |
| **Within-pack redundancy** | % inside a single pack (sloppiness within one maintainer's work) |
| **Subsumption rate** | % of nodes certified as specializations of a more general node |
| **Compression potential** | ⚠ see the transitivity warning below — **do not compute from connected components** |
| **Transitivity-violation rate** | fraction of certified triples where A≈B, B≈C, but A≉C |
| **Defect-type correspondence** | which of the paper's 7 synthetic defect types are observed in the wild, and at what rate |

### ⚠ Compression potential and the transitivity trap

> **Added 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §11.

`PERCEPTUAL` equivalence is a **threshold on a metric** (LPIPS < `τ`), and thresholded metrics are **not transitive**: `d(A,B) < τ` and `d(B,C) < τ` does not imply `d(A,C) < τ`.

So computing compression potential by taking **connected components** of the merge graph is invalid — it can collapse a chain `A≈B≈C` into one skill when `A` and `C` are not equivalent. That is a silent merge produced by the *analysis*, in a paper whose central claim is that it produces none.

**Correct procedure:** simulate the actual sequential curation policy —
1. propose merges in a defined order (e.g. by ascending distance)
2. apply one merge
3. **re-verify** the surviving canonical skill against remaining candidates
4. repeat until no certified merge remains

Report the resulting compression, and report the **transitivity-violation rate** separately. A high violation rate is a genuinely interesting finding about metric-based equivalence and strengthens the paper's case for re-verification gating.

That last row is the direct answer to R3's *"It is difficult to judge whether these corruption types reflect what actually happens in practice."* **Map each observed real defect onto one of the seven injected types, and honestly report any real defect type the taxonomy does not cover.** Discovering an eighth real defect type would be a genuine contribution — do not suppress it to protect the existing taxonomy.

---

## 8. Deliverables

| ID | Artifact |
|---|---|
| **T1** | Corpus statistics + **exclusion funnel** with reasons (§2, §6) |
| **T5** | Redundancy census (§7) |
| **T4b** | Verifier precision per relation on human-adjudicated Corpus B pairs |
| **F2** | Redundancy distribution across packs / operator families |
| **A-3** | Supplementary: frozen pack list with commit SHAs |
| **A-4** | Supplementary: full exclusion log |

**Target headline sentence:**
> *"Across M image-to-image nodes drawn from the K most-installed ComfyUI packs, VisCurate certifies Z% as redundant with an existing node, at a human-adjudicated false-merge rate of W%."*

---

## 9. Checklist

- [ ] Pull registry pack list; rank by installs; select top 50–100
- [ ] **Pin commit SHAs**; freeze and commit the pack list
- [ ] Build headless extraction shim (bypass graph executor)
- [ ] Implement the three-clause inclusion filter; record what each clause excludes
- [ ] Resolve environment; log all version conflicts
- [ ] Run determinism check; classify stochastic vs. defective
- [ ] Extend `φ`; log every conversion and type-contract violation
- [ ] Produce the exclusion funnel table
- [ ] Compute signatures (instrumented for A4 timing)
- [ ] Fingerprint → candidates → verify at frozen operating point
- [ ] Redundancy census incl. cross-pack vs. within-pack split
- [ ] **Map observed defects onto the 7 synthetic types; report uncovered types honestly**
- [ ] Launch stratified annotation batch (see [`03_annotation_and_metrics.md`](03_annotation_and_metrics.md))

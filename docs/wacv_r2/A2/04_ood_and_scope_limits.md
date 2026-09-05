# A2 · OOD Divergence and Scope Limits

**Answers:** R2's four structural critiques (Borderline Accept, confidence 4) — the reviewer most likely to be moved to a clear accept.

R2's objections are correct as stated. Arguing against them loses. **Measuring them wins**, and converts each from an unanswerable limitation into a bounded, reported quantity.

| R2 critique | Response | Where |
|---|---|---|
| *"Different skills may have same in-D performance but different OOD performance"* | Measure the certificate-flip rate on an OOD battery | §1 here |
| *"Malicious skills may appear similar but carry harmful payloads and side effects"* | Add a side-effect channel; measure what it catches | §2 here |
| *"Two skills may have same output but require vastly different token budgets"* | Non-behavioral tie-breakers | [`../A4/02_analysis_and_deliverables.md`](../A4/02_analysis_and_deliverables.md) |
| *"One skill may be written much better… for maintainability"* | Same | [`../A4/02_analysis_and_deliverables.md`](../A4/02_analysis_and_deliverables.md) |

---

## 1. OOD divergence stress test

### 1.1 The objection
R2: *"Since skills are compared over a finite synthetically constructed test suite, it is hard to judge if they will generalize to new realistic inputs. **This is a fundamental flaw in any output based approach.**"*

The paper currently concedes this qualitatively (`sec/8_discussion.tex:9`: *"a finite, probe-based approximation of equivalence, not a proof over all images"*). A concession without a number is weak. **Turn it into a measurement.**

### 1.2 Design
Build `B_ood` from distributions deliberately disjoint from the calibration battery:

| Class | Rationale |
|---|---|
| Medical imagery (X-ray, histology) | different intensity statistics, often 16-bit |
| Satellite / aerial | multispectral-like statistics, unusual textures |
| Document scans / text | high-frequency binary-ish content |
| Extreme aspect ratios (panorama, 1×N) | breaks geometric assumptions |
| Very high / very low dynamic range | exposes precision-sensitive ops |
| **Adversarial boundary probes** | kernel size ≈ image size, rotation ≈ 90°, empty masks, single-channel-constant, saturated inputs |

That last class is the highest-yield and is cheap to construct: it targets *known* operator boundary conditions rather than sampling blindly.

### 1.3 The headline metric

> **Certificate-flip rate:** of all pairs certified `EXACT` or `PERCEPTUAL` on `B`, what fraction diverge beyond tolerance on `B_ood`?

Report also:
- flip rate broken down by relation and by operator family
- flip rate by OOD class — *which* distribution breaks certificates

### 1.3a ⚠ OOD evidence is one-directional

> **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §23. The original proposed a "reverse direction" metric that is **logically confused** and has been removed.

Equivalence is **universally quantified** over the battery. Therefore:

| Observation | Valid inference |
|---|---|
| Certified equivalent on `B`, diverges on `B_ood` | **The certificate is refuted.** A counterexample defeats a universal claim. |
| Called `DISTINCT` on `B`, looks equivalent on `B_ood` | **Nothing.** The divergence observed on `B` still stands and still refutes equivalence. |

The second row is not evidence of a missed equivalence — it is just a subset on which two skills happen to agree. Reporting it as a "reverse flip" would imply the `DISTINCT` verdict was wrong, which it was not.

**Report only the refutation direction.** If a per-class agreement statistic is genuinely interesting, describe it as *"the OOD subset on which the pair does not diverge"* — never as a flipped verdict.

### 1.4 Either outcome is a good result
- **Low flip rate** → strong evidence the finite battery generalizes; rebuts R2's "fundamental flaw" framing with data
- **High flip rate** → an important honest finding, and the fix is immediate: fold the OOD classes into the default battery

### 1.5 ⚠ If the battery is extended, the first study becomes discovery

**Do not tune the battery to minimize the flip rate and then report it as if the battery were fixed in advance.** If OOD failures motivate extending `B`:

1. Label the first study explicitly as **discovery** — it found the weakness
2. Extend the battery
3. **Evaluate the revised battery on a fresh, untouched OOD holdout** built before the extension and never inspected during it
4. Report both the discovery result and the holdout result

Reporting only the post-extension number would be selection on the outcome.

---

## 2. Side-effect / security channel

### 2.1 The objection
R2: *"Malicious skills may on surface appear similar but may carry harmful payloads and side effects that do not show up when only observing output."*

Correct, and unfixable within output comparison. **So do not try to fix it there — add an orthogonal channel and measure the coverage it adds.**

### 2.2 New defect type (8th): `trojan`
Extend the corruption factory with a defect that produces **byte-identical outputs** to a donor skill while performing a side effect:

| Variant | Side effect |
|---|---|
| `trojan-fs` | writes a file outside the sandboxed temp dir |
| `trojan-net` | opens a network socket |
| `trojan-proc` | spawns a subprocess |
| `trojan-env` | reads environment variables / credentials |
| `trojan-cpu` | anomalous CPU or memory consumption |

All five keep `f_s` output-identical to the donor, so the output-grounded verifier **must** certify `EXACT`. That is the point.

Implement it as a replayable function factory matching the existing seven (`sec/6_methodology.tex:144`), so it inherits the corruption log, ground-truth derivation, and reproducibility machinery for free.

### 2.3 Instrumentation — and the containment/observability distinction

> ⚠ **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §24. This matters **operationally**, not only for the paper: the experiment deliberately executes code designed to have harmful side effects.

**`sys.addaudithook` is observability, not containment.** It reports what Python-level operations occurred; it does not prevent them, and it does not see native-code side effects at all. The original plan described it as an extension of the "hardened sandbox," conflating the two roles.

| Layer | Mechanism | Role |
|---|---|---|
| **Containment** (required first) | OS-level isolation: container/VM, **network denied by default**, no credentials or cloud metadata reachable, ephemeral filesystem discarded per run, CPU/memory/wall limits, non-root, read-only mounts except a scratch dir | prevents harm |
| **Observability** (the measurement) | `sys.addaudithook` for Python-level events; `seccomp`/`ptrace` for syscall-level including native code | records what was attempted |

**Write a short threat model** in the appendix: what the trojan variants are permitted to attempt, what the containment layer guarantees, and what is out of scope (e.g. kernel escapes). A safety-first paper that executes deliberately malicious code without stating its containment model invites an obvious objection.

**Do not present audit hooks as a security boundary** anywhere in the paper.

Record per skill a **side-effect signature** — the set of operation classes attempted during execution across the probe battery. Note that under containment, attempts may be *blocked*; record the attempt, not merely the success.

### 2.4 The result to report

| Channel | Trojan detected | Notes |
|---|---|---|
| Output-grounded verifier alone | **~0%** | outputs are identical by construction — this is the honest admission |
| + side-effect signature | **X%** | the contribution |
| False-positive rate on benign skills | Y% | some legitimate skills read files or use temp dirs |

That false-positive row matters. A side-effect channel that flags every skill using a temp file is not useful. Report it, and define the benign baseline from the clean library `L_0`.

### 2.5 Framing
Write it as:

> *Output equivalence is necessary but not sufficient for safe consolidation. We make this precise by constructing output-identical skills with divergent side effects, confirming that output grounding alone certifies them as `EXACT`, and adding an orthogonal execution-trace channel that separates them at rate X.*

**This turns R2's structural criticism into a contribution** — the strongest possible response to a Borderline-Accept reviewer, because it shows the objection was taken seriously enough to build for.

### 2.6 Gate integration
Extend the relation→action policy: a certified `EXACT` pair whose **side-effect signatures differ** must not auto-merge. Route to `UNCERTAIN` / human review. This is a one-line policy change with a clear safety rationale, and it makes the channel actionable rather than merely diagnostic.

---

## 3. Deliverables

| ID | Artifact |
|---|---|
| **T17** | Certificate-flip rate on `B_ood`, by relation / family / OOD class |
| **T18** | Trojan detection: output-only vs. + side-effect channel, with benign false-positive rate |
| **F7** | Flip rate vs. OOD class — which distributions break certificates |

### Text edits this unlocks
- `sec/8_discussion.tex:9` — the "finite, probe-based approximation" concession gains a number
- `sec/6_methodology.tex:144` — seven defect types becomes eight
- `sec/appendix.tex` §Output-Gated Curation — document the side-effect gate extension
- Limitations — R2's four critiques become three measured and one scoped

---

## 4. Checklist

- [ ] Build `B_ood` across six classes, incl. adversarial boundary probes
- [ ] Compute certificate-flip rate; break down by relation, family, OOD class
- [ ] Report the reverse direction (`DISTINCT` → equivalent) too
- [ ] **If extending the default battery in response, report pre- and post- numbers**
- [ ] Implement `trojan` defect factory with 5 variants
- [ ] Add `sys.addaudithook` side-effect recording to the sandbox
- [ ] Measure trojan detection + **benign false-positive rate**
- [ ] Extend the gate: differing side-effect signatures block auto-merge
- [ ] Update discussion, methodology, and appendix text

# A2 · Ablations — Probe Battery, Backends, Thresholds, Seeds

**Answers:** R3 minor — *"There is no analysis of the probe battery. The method depends heavily on the chosen probes, yet there is no ablation showing how performance changes with the number or diversity of probes."*

Also supplies A4 with its single biggest cost lever.

---

## 1. Probe-battery ablation ★ do this one first

The method's entire claim is *relative to a fixed probe battery* (`sec/1_intro.tex:17`). An unanalyzed battery is an unanalyzed dependency.

> ⚠ **Two corrections, 2026-08-18** — [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §21.
>
> **(a) Probe selection must be learned out-of-sample.** Choosing probes (especially by greedy diversity maximization, §1.2) using the same skill pairs the battery is then evaluated on **leaks**. Select probes on **calibration families**; evaluate on **disjoint** families — mirroring the cluster-disjoint discipline already used for thresholds (`sec/appendix.tex` §Threshold Calibration).
>
> **(b) The 256-probe point exceeds the existing battery.** `B` has 177 probes; the original sweep included 256 with no stated source for the extra 79. Either **cap the sweep at 177**, or **specify and freeze** how the battery is extended — and if it is extended, the extension must be authored before any result is seen.

### 1.1 Size sweep
`|B| ∈ {8, 16, 32, 64, 128, 177}` — 177 is the current battery. Add 256 **only** if a pre-registered extension protocol exists (see note (b) above).

For each size, report:

| Metric | Meaning |
|---|---|
| **Certificate-flip rate** vs. the full 177-probe battery | how many decisions change — the headline |
| Per-relation F1 | quality degradation by relation |
| False-merge rate | does a smaller battery admit unsafe merges? |
| Abstention rate | does it push more pairs to `UNCERTAIN`? |
| Wall-clock per pair | **→ hand this to A4** |

**Find and report the knee.** If it sits at ~64 probes, that is a 2.8× cost reduction reportable in A4.

### 1.2 Selection-strategy comparison
At each size, compare:
- **random** sampling
- **stratified by operator family**
- **greedy diversity-maximizing** — iteratively pick the probe maximizing spread in signature space

If diversity-maximizing beats random at equal size, that is a genuine method contribution: *the battery can be compressed without losing discriminative power*, and it directly addresses R1's scalability concern.

### 1.3 Probe-class ablation
Drop one class at a time and measure the flip rate:

| Class dropped | Hypothesis |
|---|---|
| Degenerate cases (1×1, all-black, all-white, single-pixel-nonzero) | expected to matter a lot — boundary conditions expose divergence |
| RGBA / alpha-channel probes | matters for compositing ops |
| 16-bit / high-dynamic-range | matters for precision-sensitive ops |
| Non-square aspect ratios | matters for geometric ops |
| Natural images (keep synthetic only) | ? |
| Synthetic patterns (keep natural only) | ? |

**Reporting which probe classes carry the discriminative signal is a real finding**, and it tells a practitioner how to build a battery for their own library — a practical contribution the paper currently lacks.

---

## 2. Backend ablation

The taxonomy uses pluggable learned backends (`sec/6_methodology.tex:72`): LPIPS for perceptual, DINO + CLIP for semantic. None is ablated.

| Configuration | Question answered |
|---|---|
| Pixel-only (`L∞`, no learned backends) | how much do learned backends buy? |
| LPIPS only | perceptual stage alone |
| DINO only | semantic stage alone |
| CLIP only | is CLIP redundant with DINO? |
| DINO + CLIP (current) | baseline |
| **Full stack (current)** | reference |

Report per-relation F1, false-merge rate, and **per-pair wall clock** for each — the last column tells A4 whether the learned stack is worth its cost, and might reveal that a cheaper configuration is nearly as good.

Also worth testing: alternative LPIPS backbones (`alex` is current; try `vgg`) and an alternative semantic encoder (e.g. a DINOv2 variant). If verdicts are stable across backbones, that is a robustness result worth one sentence.

---

## 3. Threshold sensitivity

Each threshold in Table 4 is a single calibrated value. Sweep each while holding the others fixed:

| Threshold | Current | Sweep range |
|---|---|---|
| `ε` (exact `L∞`) | 1/255 | `[1/1020, 4/255]` |
| `τ_pc` (LPIPS) | 0.05 | `[0.01, 0.20]` |
| SSIM floor (1−SSIM) | 0.10 | `[0.02, 0.30]` |
| `τ_sm` (DINO p90) | 0.15 | `[0.02, 0.30]` |
| `δ` (abstention band) | 0.05 | `[0.00, 0.15]` |

Report F1 and false-merge rate vs. each. **The `δ` sweep is the most interesting**: it directly trades abstention against decisiveness, and `δ = 0` shows exactly what the abstention band buys — a number the paper currently asserts qualitatively but never quantifies.

### 3.1 One-at-a-time sweeps miss interactions

> **Added 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §22.

The classifier is a **hierarchical cascade**, so the thresholds are not independent: `τ_pc` determines which pairs even reach the semantic stage, so its value changes the effective behavior of `τ_sm`. Sweeping each in isolation cannot reveal that.

Add a **joint sensitivity study** over at least the interacting pair `(τ_pc, τ_sm)` — a coarse 2-D grid is sufficient — and report whether the chosen operating point sits in a stable region or on a ridge. A calibrated point that is only good at one exact combination is fragile, and better to discover that ourselves.

---

## 4. Seed / calibration stability

`sec/6_methodology.tex:160` reports everything from *"a single deterministic run (seed 1234)."* One seed is not a measurement.

- Re-fit thresholds on **≥5 calibration splits** with different seeds
- Report the **operating point mean ± std** — do the calibrated thresholds move?
- Report **downstream F1 and false-merge rate mean ± std**
- If thresholds are stable across seeds, that is a strong robustness claim; if they move a lot, that is important to know before a reviewer finds it

The calibration protocol is a stated strength of the paper (`sec/appendix.tex` §Threshold Calibration: *"forecloses the dominant source of leakage"*). Demonstrating it is seed-stable converts a claim into evidence.

---

## 5. Deliverables

| ID | Artifact |
|---|---|
| **F5** | Certificate-flip rate + per-relation F1 vs. `\|B\|`, three selection strategies, knee marked |
| **T14** | Probe-class ablation — flip rate per dropped class |
| **T15** | Backend ablation — F1, FM, wall-clock per configuration |
| **F6** | Threshold sensitivity — F1 and FM vs. each of the five thresholds |
| **T16** | Seed stability — operating point and downstream metrics, mean ± std over 5 seeds |

**Numbers to hand to A4:**
- battery knee size → cost reduction factor
- per-pair wall clock per backend configuration → the cheapest adequate stack

---

## 6. Checklist

- [ ] Size sweep `|B| ∈ {8…256}`; report flip rate; **find the knee**
- [ ] Three selection strategies at each size
- [ ] Probe-class drop-one ablation
- [ ] Backend ablation incl. per-pair wall clock
- [ ] Alternative backbones (LPIPS-vgg, DINOv2 variant)
- [ ] Five threshold sweeps; emphasize `δ`
- [ ] Re-calibrate on 5 seeds; report mean ± std
- [ ] **Hand the knee size and per-config timing to A4**

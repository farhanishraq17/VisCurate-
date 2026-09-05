# VisCurate — Next Actions (decisions made, work order)

**Date:** 2026-08-19
**Context:** Follows the Stage 0–5 completion report. Stages 0, 1, 2, 3, 4, 5 landed; `G_ρ` replay did not.
**Reference docs:** [`RUNBOOK.md`](RUNBOOK.md), [`CODEBASE_RECONCILIATION.md`](CODEBASE_RECONCILIATION.md), [`A1/`](A1/)–[`A4/`](A4/)

First: the Stage 1 result is the big one. The full 944-pair divergence run passing go/no-go resolves the `phase4.md` vs. `WORK_PROCESS_OVERVIEW.md` contradiction that was blocking everything. Thank you for flagging your own gap on the `G_ρ` replay rather than letting it slide — that's the right call and it's item 1 below.

**One check before anything else:** compare the new Stage-1 numbers against what `phase4.md` claimed. If they differ, **the paper follows the new run.** Report both so the discrepancy is on record.

---

## Decisions — all four now settled

| # | Question | Decision |
|---|---|---|
| 1 | Hardened sandbox? | **Yes — but via a disposable VM, not by building one.** See §4. |
| 2 | Annotators? | **Yes for the 26 adjudications, immediately, in-house.** Broader κ is separate and non-blocking. |
| 3 | A3 scope `[CONFIRM]`? | **Scoped feasibility study only.** Segmentation + depth. Last priority, first to cut. |
| 4 | GPT-5.5 at 259/300? | **Finish the 41 episodes.** |

---

## Priority order

| # | Task | Effort | Blocks |
|---|---|---|---|
| **1** | `G_ρ` replay | hours, no GPU | R3's verifier-validation critique |
| **2** | The 26 label adjudications | ~1 hour | the 4.25% vs 6.73% safety bound |
| **3** | GPT-5.5's remaining 41 episodes | small run | Table 1's headline number |
| **4** | Corpus B on a disposable VM | 1–2 days | AC required revision #1 |
| **5** | A3 feasibility version | 1–2 weeks | AC required revision #3 |
| — | Broader κ slice | parallel | not blocking |

Items 1–3 are cheap and each upgrades something load-bearing. Do them first.

---

## 1. `G_ρ` replay — run it

Your own diagnosis was right: `G_0` is a clean library, so its mergeable class is empty by construction, and merge recall — accuracy on the only two relations that authorize deleting a skill — is unmeasurable there. `G_ρ` is the only set with `EXACT`/`PERCEPTUAL` support. This is R3's *"the verifier itself is not validated"* complaint, and right now the verifier is validated on everything **except** the two relations that can destroy a capability.

**Before running — verify the answer key is complete.** Do not skip this because the replay is cheap. The corruption log records the relations each defect *injects*, but may not record **incidental** relations the injection creates (e.g. inserting a duplicate of skill `S` also creates relations between that duplicate and everything `S` was already related to). If incidental relations are unlabelled and the evaluation treats unlabelled pairs as `DISTINCT` by default, **the ground truth is wrong and the verifier gets penalised for correct verdicts.**

Check a sample of instances by hand. Then either materialise the transitive closure, or restrict evaluation to explicitly-labelled pairs — and **say which you did.**

**Report:**
- Per-relation P/R/F1 **with support counts** — especially `EXACT` and `PERCEPTUAL`
- The binary *mergeable* decision, now with a populated positive class
- Full 7×7 confusion matrix including `UNCERTAIN`
- Breakdown by corruption rate ρ
- **Bootstrap by corruption instance / base skill / seed — never by pair.** Pairs within an instance share skills and are strongly dependent; pair-level bootstrap badly understates variance.
- Exact (Clopper–Pearson) intervals on every rate, with raw counts

---

## 2. The 26 adjudications — do these this week

The hard-negative slice is 0/69, but 26 of those 69 need human label confirmation:

| Outcome | n | 95% upper bound |
|---|---|---|
| 26 confirmed | 69 | **4.25%** |
| 26 rejected | 43 | **6.73%** |

**4.25% clears the ≤5% credibility threshold. 6.73% does not.** These 26 pairs decide whether the headline safety claim is defensible or merely suggestive.

**Protocol:**
- **3 annotators**, blind to the verifier's verdict and to each other
- Show only: both outputs side by side on 8–12 probes **at the divergent parameter setting**, plus a difference map and both docstrings
- Label set is the six relations plus **`CANNOT-TELL`** — the escape hatch is essential; forcing a guess manufactures agreement and corrupts κ
- Adjudicate disagreements by discussion; no consensus → `CANNOT-TELL`, reported separately
- Compute κ even on this small set, and report the `CANNOT-TELL` rate

**Report both bounds regardless of outcome** — 4.25% conditional on the adjudications, 6.73% unconditional. Do not report only the favourable one.

Co-authors can do this set. The larger SEMANTIC/SUBSUMPTION κ slice can use external annotators later and is not blocking.

---

## 3. GPT-5.5 — finish the 41

The abstract's central claim (*"the strongest curator recovers only about half of the required repairs, recall 0.538, Mean F1 0.583"*) is GPT-5.5's number. Every other model has n=300; GPT-5.5 has 259. That asymmetry is the first thing a careful reviewer checks.

Run the remaining 41. If for any reason they can't be run, **report n=259 explicitly in the table and say why** — do not silently pool it with n=300 rows.

---

## 4. Corpus B — the disposable-VM approach

Corpus A is done and is a strong result, but **it does not satisfy the AC's literal ask.** The AC requested *"at least one real, **uncurated** open-source skill library."* OpenCV, Kornia, Albumentations, and torchvision are **curated** — they have maintainers and review processes. Corpus B (ComfyUI community plugins) is the only corpus that literally answers the request.

**Do not build a hardened sandbox.** Use a **disposable cloud VM** instead: containment through disposability rather than through a hardened runtime. Cheaper, faster, and — for this specific job — safer than a hand-rolled sandbox, because a bad sandbox is worse than no sandbox (it gives false confidence).

Most of Corpus B is CPU-only: the inclusion filter accepts only nodes whose required inputs are an `IMAGE` plus primitives (`INT`/`FLOAT`/`STRING`/`BOOLEAN`/`COMBO`), which excludes anything needing `MODEL`/`VAE`/`CLIP`/`LATENT`. So a cheap CPU box is sufficient.

---

### ⚠ SECURITY — read this section fully before touching any community plugin

**You will be downloading and executing arbitrary Python written by strangers, with no code review, no signing, and no accountability.** Treat every pack as hostile by default. This is not a hypothetical: plugin ecosystems with no gatekeeping are a known malware distribution channel.

#### The non-obvious risk: extraction is already execution

There is **no safe inspection phase.** Enumerating a pack's nodes requires importing it, and `__init__.py` runs on import — *before* you call any skill function. **Merely listing what's in a pack executes that pack's code.** Plan containment from the very first step, not from the point where you start calling functions.

#### Required containment

| Control | Requirement |
|---|---|
| **Machine** | Dedicated, disposable cloud VM. **Destroy it when done.** Never reuse it for anything else. |
| **Credentials** | **Nothing valuable on the box.** No SSH keys, cloud creds, API keys, git tokens, `.env` files, password-manager data, browser profiles. |
| **Cloud metadata** | **Block `169.254.169.254`.** Cloud instance-metadata is a classic credential-exfiltration path — malicious code reads it to steal the VM's own IAM role. |
| **Network** | Egress **denied by default**. Open only to PyPI/GitHub during install, then **cut network entirely** for the execution phase. Skills are image→image; they have no legitimate reason to touch the network. |
| **User** | Non-root. No sudo. |
| **Filesystem** | Ephemeral. No mounts of anything you care about. Snapshot before, destroy after. |
| **Limits** | Per-call CPU, memory, and hard wall-clock timeout. Kill on breach. |
| **Provenance** | **Pin commit SHAs** for every pack. Required for reproducibility anyway; doubles as supply-chain hygiene against a repo being altered mid-experiment. |

#### Pre-flight triage (not a security boundary — a filter and a finding)

Before executing, static-scan each pack for: `eval`, `exec`, `__import__`, `subprocess`, `os.system`, `socket`, `requests`, `urllib`, base64 blobs, obfuscated strings, writes outside a temp dir, and install-time hooks in `setup.py`/`pyproject.toml`.

**Anything flagged: quarantine it and look at it manually before running.**

Note this is **also data for the paper.** A community image-processing plugin that opens a socket is a genuine finding about real skill-library hygiene — log it, count it, report it. That's R2's security critique with real evidence behind it.

#### Explicitly do not

- ❌ **Do not run this on the GPU cluster.** Untrusted code has no business on a shared machine with other people's work on it.
- ❌ Do not run on a laptop or any machine with personal data.
- ❌ Do not run as root.
- ❌ Do not reuse the VM afterward.
- ❌ **Do not treat `sys.addaudithook` as containment.** Audit hooks *observe* Python-level events; they do not *prevent* anything, and they see nothing of native code. Use them for the side-effect telemetry channel, never as the security boundary.

**If anything looks actively malicious — obfuscated payloads, credential access, C2-style callbacks — stop, preserve the sample, and report it before continuing.** Don't quietly work around it.

---

### Corpus B experimental protocol

Beyond the security work, follow [`A1/02_corpus_B_comfyui.md`](A1/02_corpus_B_comfyui.md). The three things most likely to be got wrong:

1. **Stratified sampling, two estimands.** Take a **popular** stratum (top-K by installs) *and* a **random long-tail** stratum with a recorded seed. Report them **separately**. Never write "X% of ComfyUI nodes are redundant" from the popular stratum alone — that measures deployments, not the ecosystem.
2. **Isolated environment per pack**, not one shared env. A shared env produces order-dependent dependency conflicts, which makes your exclusion funnel partly an artifact of the harness — and the exclusion funnel is a headline result.
3. **Contract violations fail the gate; do not repair them.** A node returning `uint8` when it declared `IMAGE`, or values outside `[0,1]`, or `BCHW` instead of `BHWC` — **exclude it and log the violation.** Do **not** clamp values or guess the layout: that can make two genuinely different nodes look identical, which manufactures a false merge. Type-contract violations are themselves a finding about library decay.

Also: **do not compute compression potential from connected components** of the merge graph. `PERCEPTUAL` equivalence is a threshold on a metric and is **not transitive** — A≈B and B≈C does not give A≈C. Simulate the actual sequential curation policy with re-verification after each merge, and report the transitivity-violation rate separately.

---

## 5. A3 — scoped feasibility study only

The `[CONFIRM]` scope line (*"No generative/diffusion skills in v1"*) is hereby resolved as: **partially reopened, as a feasibility study, not a full study.**

**Scope:** segmentation + depth only (~8 operators — SAM 2 variants, MobileSAM, Depth Anything V2 variants, MiDaS). These have the cleanest comparators (mask IoU; affine-aligned depth error) and both appear by name in R1's critique.

**What A3 must NOT claim:** that the 0.99 non-equivalence precision guarantee transfers. There isn't enough sample support. Report observed counts with exact bounds and state the limit plainly. *"No false merges were observed among N hard negatives (95% upper bound X%)"* is true and checkable; *"safety preserved"* is not.

**Two mandatory technical points** (see [`A3/02_method_distributional_equivalence.md`](A3/02_method_distributional_equivalence.md)):

- **Compare conditionally, per probe — never pool across probes.** Pooling all probes and seeds into one bag per skill destroys conditioning on the input: skill A doing `blur(x₁), sharpen(x₂)` and skill B doing `sharpen(x₁), blur(x₂)` have *identical* pooled distributions and would certify as equivalent while behaving completely differently. That is a false merge by construction. Compute the discrepancy per `(probe, parameter)` then aggregate by worst case — the same rule the deterministic path already uses.
- **Use a permutation null, not a naive bootstrap CI.** The unbiased MMD estimator can be negative and has a degenerate distribution under equality — exactly the regime where equivalence decisions are made. Validate coverage in simulation before trusting it.

**A3 is the correct thing to cut if the deadline tightens.** If cut, that must be an explicit sentence in the rebuttal, not a silent omission.

---

## Rules that still hold

1. **Never re-calibrate thresholds on evaluation data.** Carry over the frozen operating point unchanged. No test-driven repair.
2. **Freeze and hash before running** — pair lists, pack lists with SHAs, alignment maps, annotation guidelines.
3. **Every rate gets raw counts and an exact interval.** `0/69` is not a result; `0/69 [0, 4.25%]` is.
4. **Never fabricate.** "Not yet run" is an acceptable answer. Keep reporting blocked / deliberately-not-done / partial as separate categories — that distinction in your last report was genuinely useful.
5. **Report numbers exactly as produced.** If they disagree with the paper draft, the draft is what changes.

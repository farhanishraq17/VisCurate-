# VisCurate — pipeline diagrams (Mermaid)

Mermaid source for the whole project pipeline. Three views: the end-to-end data + method
flow (the spine), the output-grounded verifier's stop-at-first taxonomy, and the phase
roadmap. Renders on GitHub and at <https://mermaid.live>.

---

## 1. End-to-end pipeline (data + method flow — the spine)

```mermaid
flowchart TD
    %% ---------------- Dataset construction (5 layers) ----------------
    subgraph DATA["Dataset construction — family indexed by (ρ, composition c, seed)"]
        direction TB
        A["Layer A — Audited primitive ops<br/>(trusted atoms behind each skill.fn)"]
        B["Layer B — Clean base library L0 — 100 skills<br/>→ designed relation graph G0"]
        C["Layer C — Probe battery P<br/>versioned, license-clean, + parameter sweeps"]
        ORACLE["Frozen reference oracle<br/>hashes of every L0 output over P<br/>scores/confirms — never assigns labels"]
        DEF["7 defect injectors<br/>bug · metadata · duplicate · subsumption<br/>param-schema · domain-scoped · dead-skill"]
        D["Layer D — Corrupted library L_ρ<br/>→ G_ρ + corruption log + ideal-action key"]
        E["Layer E — Task / usage layer T<br/>query→skill relevance + Zipfian usage log"]
    end

    A --> B
    B --> ORACLE
    C --> ORACLE
    B --> D
    DEF -->|inject at rate ρ| D
    C --> D
    D --> E

    %% ---------------- The architectural split ----------------
    subgraph SPLIT["The architectural split (the single most important decision)"]
        direction LR
        VER["VERIFIER — outputs only, text-blind<br/>what IS the relation?"]
        AGENT["AGENT — repository decisions<br/>what SHOULD be done?"]
    end

    C --> VER
    D --> VER
    E --> AGENT
    VER -->|certifying relation gates merge/split/parameterize| AGENT
    AGENT -->|structured rejection + reason| VER

    AGENT --> CUR["Curation actions<br/>add · remove · modify · retrieve<br/>merge · split · parameterize · end"]

    %% ---------------- Downstream + studies ----------------
    CUR --> DOWN["Downstream solver agent<br/>held-out queries, disjoint from probes<br/>perceptual match + task predicates"]
    ORACLE --> DOWN

    subgraph ST["Four studies (mean ± CI across seeds)"]
        direction TB
        S1["Study 1 — Equivalence F1<br/>per judge track: name / embedding /<br/>LLM-on-text / output-grounded"]
        S2["Study 2 — Curation Pareto<br/>success ↑ × compression × action cost"]
        S3["Study 3 — Construct validity<br/>intrinsic score vs downstream success"]
        S4["Study 4 — Vision-matters ablation<br/>text-cosine gating vs output gating"]
    end

    DOWN --> S1
    DOWN --> S2
    DOWN --> S3
    DOWN --> S4
    S1 --> PAPER["Paper artifacts<br/>every number ← a committed run manifest"]
    S2 --> PAPER
    S3 --> PAPER
    S4 --> PAPER
```

---

## 2. The output-grounded verifier (stop-at-first taxonomy)

```mermaid
flowchart TD
    START["classify(A, B)<br/>compare over matched sweep × probes<br/>text-blind: ComparatorView + outputs only"]
    START --> CAND{"output-based candidate?<br/>fingerprint NN ∪ same-family ∪ hard negatives"}
    CAND -->|no| DIST
    CAND -->|yes| EX

    EX{"EXACT?<br/>hash match or worst-case L∞ ≤ ε"}
    EX -->|yes| REX["EXACT → merge"]
    EX -->|no| PE

    PE{"PERCEPTUAL?<br/>worst-case LPIPS ≤ τ_perc + SSIM floor"}
    PE -->|in band τ·1±δ| UNC["UNCERTAIN → human review"]
    PE -->|yes| RPE["PERCEPTUAL → merge after validation"]
    PE -->|no| SUB

    SUB{"SUBSUMPTION?<br/>directional grid search, A ⊑ B"}
    SUB -->|yes| RSUB["SUBSUMPTION → parameterize"]
    SUB -->|no| SEM

    SEM{"SEMANTIC?<br/>DINO p90 ≤ τ_sem, CLIP optional"}
    SEM -->|in band| UNC
    SEM -->|yes| RSEM["SEMANTIC_PRESERVING → unify or keep both"]
    SEM -->|no| COMP

    COMP{"COMPLEMENTARY?<br/>non-trivial + approximate commutation"}
    COMP -->|yes| RCOMP["COMPLEMENTARY → keep both"]
    COMP -->|no| DIST["DISTINCT → reject merge"]
```

---

## 3. Phase roadmap (build order + status)

```mermaid
flowchart LR
    P0["Phase 0<br/>Scaffold ✅"] --> P1["Phase 1<br/>Skills + harness ✅"]
    P1 --> P2["Phase 2<br/>Probe battery + oracle ✅"]
    P2 --> P3["Phase 3<br/>Comparators + taxonomy ✅"]
    P3 --> P4{"Phase 4 ⚠️<br/>Equivalence benchmark<br/>divergence go/no-go"}
    P4 -->|divergence confirmed| P5["Phase 5<br/>Corruption generator"]
    P5 --> P6["Phase 6<br/>Curation environment<br/>+ hardened sandbox"]
    P6 --> P7["Phase 7<br/>Query stream + downstream eval"]
    P7 --> P8["Phase 8<br/>Metrics, baselines, studies"]
    P8 --> P9["Phase 9<br/>Experiment runner + paper artifacts"]
    P4 -.->|no divergence: stop & analyze| STOP["Re-examine premise"]
```
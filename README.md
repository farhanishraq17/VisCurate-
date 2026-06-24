# VisCurate

Output-grounded equivalence verification and automated curation for visual skill
libraries. A *skill* is a deterministic image→image function; the project asks whether
two skills are equivalent **by executing them and comparing outputs**, not by comparing
their text descriptions — and uses those output-grounded relations to gate library
curation actions (merge / split / parameterize).

See [claude.md](claude.md) for the full implementation roadmap and locked decisions.

## Status

Phase 0–9 machinery is implemented. Real paper numbers still require the Phase-4
LPIPS/DINO/CLIP divergence go/no-go run and the full study grid.

- **Phase 0 — Scaffold.** Package layout, Pydantic-validated config, explicit seeded
  RNG (no global state), structlog JSON logging, lint/type/test toolchain.
- **Phase 1 — Skills + harness (complete).** `Skill` model + JSON-serializable
  registry, the output canonicalization contract (§1.3), a lightweight sandboxed
  executor (subprocess + wall-clock timeout + `trusted` gating), and all 100 deterministic
  skills across the geometric, colour, signal, and reconstruction families.
- **Phase 2 — Probe battery + reference oracle (complete).** Deterministic license-clean
  probe generators (CC0 synthetics + license-filtered COCO CC BY photos), a coverage-checked
  manifest (no `license=unknown`), and a frozen reference oracle with determinism self-audit.
  Build with `viscurate build-probes` / `viscurate freeze-oracle`.
- **Phases 3–8 — Equivalence, corruption, curation, downstream, and study aggregation
  machinery.** Output-grounded verifier, text baselines, corruption grid, curation environment,
  query stream, downstream evaluator, and Phase-8 aggregation/reporting.
- **Phase 9 — Experiment runner + paper artifacts.** `viscurate phase9` writes the
  manifest-backed reproducibility bundle, realism audit, `reproduce.sh`, and Phase-8 paper
  tables/figures when supplied real `StudyPoint` rows.

## Architectural rule (load-bearing)

The output-grounded comparator path **must never read a skill's `description`**. Text and
embedding baselines live in a separate package that may. This is a hard boundary, not a
convention (CLAUDE.md §1.2). The ML comparators (LPIPS / DINO / CLIP) are an optional
`[ml]` extra so the skill harness stays importable with zero ML dependencies.

## Install

```bash
python -m pip install -e ".[dev]"        # Phase 0 + 1 (dependency-light)
python -m pip install -e ".[dev,ml]"     # adds Phase 3 comparators (torch, lpips, ...)
```

## Develop

```bash
ruff check .          # lint
ruff format .         # format
mypy src              # strict type-check
pytest                # tests
```

## Reproduce

```bash
viscurate phase9 -c configs/phase9.yaml -o results/phase9
```

The Phase-9 bundle records config hashes, git SHA, model package versions, artifact manifests,
and pending/missing empirical runs without inventing numbers.

## Layout

```
src/viscurate/
  config.py            # Pydantic-validated YAML config (no literals in code)
  rng.py               # explicit seed derivation, no global RNG state
  logging.py           # structlog JSON logging setup
  cli.py               # `viscurate` entry point
  skills/
    model.py           # Skill model + ParamSpec/ParamsSchema
    registry.py        # JSON-serializable registry
    canonicalize.py    # the output canonicalization contract (§1.3)
    executor.py        # sandboxed executor (subprocess + timeout + trusted gate)
    library/           # the 100 deterministic skill implementations
  probes/
    manifest.py        # Probe/Manifest + License models (no license=unknown)
    synthetics.py      # deterministic CC0 probe generators
    coco.py            # license-filtered COCO natural-photo loader
    build.py           # battery orchestrator + reproducibility manifest
    oracle.py          # frozen reference oracle (freeze + verify)
  experiments/
    config.py          # Phase-9 experiment config
    manifest.py        # run manifests + reproduce.sh
    audit.py           # realism audit over manifests/licenses/splits
    runner.py          # Phase-9 bundle writer
configs/default.yaml   # example configuration
configs/probes.yaml    # probe-battery configuration
tests/                 # determinism, serialization, executor, canonicalization, probes, oracle
```

## Reproducibility

Everything is seed-parameterized. Determinism given `(image, params, seed)` is a hard
requirement so that "same output" is decidable (CLAUDE.md §1.4). Never fabricate results:
`results/` and every figure come from a real run with a committed manifest.

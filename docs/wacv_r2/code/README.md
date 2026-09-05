# Drop-in code for the VisCurate repo

> **Status in this repo (2026-09-04).** All four files are installed at their destinations —
> `src/viscurate/instrument/telemetry.py`, `src/viscurate/baselines/embedders.py`,
> `src/viscurate/baselines/code_judges.py`, `tests/test_code_judges.py` — and the installed copies
> are the maintained ones (`telemetry.py` gained `NullRecorder` / `active_recorder`; the others differ
> from the drafts only by `ruff format`). The draft `.py` files that used to sit next to this README
> were not committed, so there is one version of each module in the tree. The table below is kept as
> the install record.

Four files. Each has a **DESTINATION** path in its module docstring — copy it there, don't import from this folder.

Target repo: `D:\Research_Ishmam\Viscurate\Serious Deadlines\VisCurate_Experiments\`

| File | Destination | Purpose | Deps |
|---|---|---|---|
| [`viscurate_instrument_telemetry.py`](viscurate_instrument_telemetry.py) | `src/viscurate/instrument/telemetry.py` | **A4 telemetry — install FIRST** | none (torch optional) |
| [`viscurate_baselines_embedders.py`](viscurate_baselines_embedders.py) | `src/viscurate/baselines/embedders.py` | A2 rung 3 (sentence-embedding) | `sentence-transformers` |
| [`viscurate_baselines_code_judges.py`](viscurate_baselines_code_judges.py) | `src/viscurate/baselines/code_judges.py` | A2 rungs 4, 5, 7 (code baselines) | `transformers` (rung 4 only) |
| [`test_code_judges.py`](test_code_judges.py) | `tests/test_code_judges.py` | tests for the above | none |

## Install

```bash
cd /path/to/VisCurate_Experiments
mkdir -p src/viscurate/instrument
cp <plans>/code/viscurate_instrument_telemetry.py   src/viscurate/instrument/telemetry.py
printf 'from viscurate.instrument.telemetry import recorder, timer, gpu_timer\n' > src/viscurate/instrument/__init__.py
cp <plans>/code/viscurate_baselines_embedders.py    src/viscurate/baselines/embedders.py
cp <plans>/code/viscurate_baselines_code_judges.py  src/viscurate/baselines/code_judges.py
cp <plans>/code/test_code_judges.py                 tests/test_code_judges.py

pytest tests/test_code_judges.py -q
ruff check src/viscurate/instrument src/viscurate/baselines tests/test_code_judges.py
mypy src/viscurate/instrument src/viscurate/baselines
```

`test_code_judges.py` needs no optional dependencies and should pass immediately.

## Status

**Syntax-checked and the AST logic is functionally verified** (2026-08-18) against the repo's real interfaces — `TextJudge` / `JudgeVerdict` / `LlmClient` protocols in `baselines/judges.py`, `Relation` in `equivalence/relations.py`, `Skill` in `skills/model.py`.

Verified AST behaviour:

```
default-drift pair:   structure-only=1.000   semantics-preserving=0.875
renamed-vars pair:    structure-only=1.000   semantics-preserving=1.000
different-ops pair:   semantics-preserving=0.400
```

The first row is the point: **structure-only is blind to a changed default parameter** (corruption defect type v) while semantics-preserving sees it. That asymmetry is why both variants must be reported — see the module docstring, rule 2.

**Not yet run against the real repo** (no Python env with `viscurate` installed on this machine). Expect minor import/lint adjustments on first integration; the logic is tested, the wiring is not.

## Three things not to "simplify" away

1. **Docstrings stripped for rung 4** — otherwise a "code embedding" is partly a description embedding and stops testing what it claims to.
2. **Both AST variants reported** — see above.
3. **Unparseable LLM replies → `UNCERTAIN`, not `DISTINCT`** — the shipped `LlmJudge` maps them to DISTINCT, which hides model failure inside a safety number. `LlmSourceJudge` deliberately differs; report the abstention rate separately.

## Also note

- `TextEmbedder` in `baselines/judges.py` is **already swappable**, so rung 3 needs no new judge — just pass `SentenceTransformerEmbedder` to the existing `EmbeddingCosineJudge`.
- Telemetry records **tokens and latency, never dollars** — prices change and a run must stay reproducible. Convert to USD at analysis time with a timestamped price table.
- `gpu_timer` uses CUDA events with an explicit `synchronize()`. Naive `time.time()` around an async CUDA call measures the launch, not the work.

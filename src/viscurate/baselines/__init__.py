"""Text / embedding / LLM baselines (CLAUDE.md §3.4) — the judges the project competes against.

This package is the **only** one allowed to read a skill's ``description``. It is deliberately
separate from :mod:`viscurate.equivalence` (which is text-blind by type), so the load-bearing
modality boundary (CLAUDE.md §1.2) is a package boundary, not a convention. The baselines:

* **name-match** — token similarity over names/ids (the trivial renamed-duplicate detector);
* **embedding-cosine** — cosine over description embeddings (the *direct strawman* the paper
  attacks); the embedder is dependency-light TF-IDF by default, behind a swappable protocol;
* **LLM-on-descriptions** — an LLM judge over descriptions (built behind a client protocol; it
  runs only when a real client is supplied — there is no offline stub that fabricates answers).

Each judge yields a :class:`JudgeVerdict` carrying a comparable binary ``mergeable`` decision
plus a similarity score, so every track is scored on the same axis as the output verifier.
"""

from __future__ import annotations

from viscurate.baselines.judges import (
    EmbeddingCosineJudge,
    JudgeVerdict,
    LlmClient,
    LlmJudge,
    LlmUnavailableError,
    NameMatchJudge,
    TextEmbedder,
    TextJudge,
    TextRecord,
    TfidfEmbedder,
    UnavailableLlmClient,
    text_record_from_spec,
)

__all__ = [
    "EmbeddingCosineJudge",
    "JudgeVerdict",
    "LlmClient",
    "LlmJudge",
    "LlmUnavailableError",
    "NameMatchJudge",
    "TextEmbedder",
    "TextJudge",
    "TextRecord",
    "TfidfEmbedder",
    "UnavailableLlmClient",
    "text_record_from_spec",
]

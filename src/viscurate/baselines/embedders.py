"""A2 rung 3 — sentence-embedding baseline (drop-in ``TextEmbedder``).

DESTINATION: ``src/viscurate/baselines/embedders.py`` in the VisCurate repo.

The existing ``EmbeddingCosineJudge`` already takes a swappable ``TextEmbedder`` protocol
(``baselines/judges.py``), so rung 3 of the A2 baseline ladder needs **no new judge** — only a
stronger embedder than the default deterministic TF-IDF. That is the entire delta:

    from viscurate.baselines.judges import EmbeddingCosineJudge
    from viscurate.baselines.embedders import SentenceTransformerEmbedder

    judge = EmbeddingCosineJudge(embedder=SentenceTransformerEmbedder(), tau=...)

Why this matters for the paper: R3's critique is that the central claim was never tested against
a real competitor. TF-IDF alone is dismissible as a strawman; a modern sentence encoder is not.

Dependencies (optional extra — keep them lazy so the package still imports without them):
    pip install sentence-transformers
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["SentenceTransformerEmbedder"]

NDArrayF = npt.NDArray[np.float32]

# Verify availability at implementation time; all three are standard on the HF hub.
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
ALTERNATIVES = (
    "thenlper/gte-large",
    "intfloat/e5-large-v2",
)


@dataclass
class SentenceTransformerEmbedder:
    """A sentence-transformer text embedder satisfying the ``TextEmbedder`` protocol.

    The model is imported and loaded **lazily** (inside ``embed``), matching the pattern in
    ``equivalence/backends.py`` so importing this module never requires the optional dependency.

    ``normalize=True`` returns L2-normalized vectors, which the protocol requires: the judge
    takes a plain dot product as cosine similarity.
    """

    model_id: str = DEFAULT_MODEL
    device: str = "cpu"
    normalize: bool = True
    name: str = field(default="", init=False)

    def __post_init__(self) -> None:
        # The protocol declares `name` as a plain attribute; derive it from the model so the
        # run manifest records *which* encoder produced the numbers.
        object.__setattr__(self, "name", f"sentence-embedding[{self.model_id.split('/')[-1]}]")
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "SentenceTransformerEmbedder needs `pip install sentence-transformers`"
                ) from exc
            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def embed(self, text: str) -> NDArrayF:
        model = self._ensure_model()
        vec = model.encode(  # type: ignore[attr-defined]
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype=np.float32)

    def close(self) -> None:
        """Free the model — mirrors the one-model-at-a-time discipline in ``backends.py``."""
        self._model = None

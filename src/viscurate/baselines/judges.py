"""The text baselines (CLAUDE.md §3.4) — judges that read names/descriptions only.

A judge maps an ordered pair of :class:`TextRecord` to a :class:`JudgeVerdict`: a binary
``mergeable`` decision (the axis comparable with the output verifier) plus a similarity score.
None of these judges ever sees an executed output — that is the whole point. The contribution
is that the output verifier *disagrees* with them on the engineered hard negatives (similar
text, different behaviour) and on the redundancy they miss (different text, identical output).
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from viscurate.equivalence.relations import Relation
from viscurate.skills.model import SkillSpec

__all__ = [
    "EmbeddingCosineJudge",
    "JudgeVerdict",
    "LlmClient",
    "LlmJudge",
    "LlmUnavailableError",
    "NameMatchJudge",
    "OpenAIClient",
    "TextEmbedder",
    "TextJudge",
    "TextRecord",
    "TfidfEmbedder",
    "UnavailableLlmClient",
    "text_record_from_spec",
]

NDArrayF = npt.NDArray[np.float32]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (``"Grayscale (BT.601)" -> [grayscale, bt, 601]``)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class TextRecord:
    """The text a baseline is allowed to read about one skill (CLAUDE.md §1.2)."""

    id: str
    name: str
    description: str
    tags: tuple[str, ...] = ()

    def text(self) -> str:
        """The concatenated text surface used by the embedding judge."""
        return " ".join((self.name, self.description, *self.tags))


def text_record_from_spec(spec: SkillSpec) -> TextRecord:
    """Project a :class:`~viscurate.skills.model.SkillSpec` to its readable text surface."""
    return TextRecord(
        id=spec.id, name=spec.name, description=spec.description, tags=(spec.metadata.family,)
    )


@dataclass(frozen=True)
class JudgeVerdict:
    """A text judge's answer for one pair.

    ``mergeable`` is the binary decision compared across tracks; ``relation`` is the coarse
    relation guess (text judges cannot resolve the fine 6-way taxonomy, so it is EXACT when
    mergeable else DISTINCT, unless a judge — e.g. the LLM — returns a finer guess);
    ``similarity`` is in ``[0, 1]`` (higher = more similar) for thresholding / ROC.
    """

    mergeable: bool
    relation: Relation
    similarity: float


@runtime_checkable
class TextJudge(Protocol):
    """A baseline that decides a pair's relation from text alone.

    ``name`` is a read-only property so the frozen-dataclass judges below satisfy the protocol.
    """

    @property
    def name(self) -> str: ...

    def verdict(self, a: TextRecord, b: TextRecord) -> JudgeVerdict: ...


# --------------------------------------------------------------------------------------------
# name-match — token similarity over names + ids (renamed-duplicate detector).
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NameMatchJudge:
    """Jaccard token overlap over ``name`` ∪ ``id`` tokens; merge iff above ``tau``."""

    tau: float = 0.5
    name: str = field(default="name-match", init=False)

    def _tokens(self, r: TextRecord) -> set[str]:
        return set(_tokenize(r.name)) | set(_tokenize(r.id))

    def verdict(self, a: TextRecord, b: TextRecord) -> JudgeVerdict:
        ta, tb = self._tokens(a), self._tokens(b)
        union = ta | tb
        sim = len(ta & tb) / len(union) if union else 0.0
        mergeable = sim >= self.tau
        return JudgeVerdict(
            mergeable=mergeable,
            relation=Relation.EXACT if mergeable else Relation.DISTINCT,
            similarity=sim,
        )


# --------------------------------------------------------------------------------------------
# embedding-cosine — the direct strawman. Dependency-light TF-IDF embedder by default.
# --------------------------------------------------------------------------------------------


@runtime_checkable
class TextEmbedder(Protocol):
    """Maps text to an L2-normalized vector; swappable (TF-IDF here, sentence-model elsewhere)."""

    name: str

    def embed(self, text: str) -> NDArrayF: ...


class TfidfEmbedder:
    """A deterministic, dependency-light TF-IDF embedder fit on the skill-text corpus.

    No external model or download: tokenize → term frequencies → smoothed IDF → L2-normalized
    vector. Reproducible by construction, which keeps the baseline auditable (CLAUDE.md §5).
    """

    def __init__(self, corpus: Sequence[str]) -> None:
        self.name = "tfidf"
        vocab: dict[str, int] = {}
        doc_freq: dict[str, int] = {}
        for doc in corpus:
            seen: set[str] = set()
            for tok in _tokenize(doc):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
                if tok not in seen:
                    doc_freq[tok] = doc_freq.get(tok, 0) + 1
                    seen.add(tok)
        n_docs = max(1, len(corpus))
        self._vocab = vocab
        # Smoothed idf (sklearn-style): ln((1+N)/(1+df)) + 1.
        self._idf = np.ones(len(vocab), dtype=np.float32)
        for tok, idx in vocab.items():
            self._idf[idx] = math.log((1.0 + n_docs) / (1.0 + doc_freq[tok])) + 1.0

    def embed(self, text: str) -> NDArrayF:
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        if not self._vocab:
            return vec
        for tok in _tokenize(text):
            idx = self._vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        vec *= self._idf
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 0 else vec


@dataclass(frozen=True)
class EmbeddingCosineJudge:
    """Cosine similarity over description embeddings; merge iff similarity ≥ ``tau``.

    This is the "description-embedding cosine dedup" baseline the project competes against
    (CLAUDE.md §3.4). It merges skills whose *descriptions* are close — which is exactly why it
    wrongly merges ``blur_gaussian``/``blur_box`` (both "convolve with a kernel") and misses
    redundancy hidden behind unrelated wording.
    """

    embedder: TextEmbedder
    tau: float = 0.6
    name: str = field(default="embedding-cosine", init=False)

    def verdict(self, a: TextRecord, b: TextRecord) -> JudgeVerdict:
        va, vb = self.embedder.embed(a.text()), self.embedder.embed(b.text())
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        sim = float(np.dot(va, vb) / (na * nb)) if na > 0 and nb > 0 else 0.0
        sim = max(0.0, min(1.0, sim))
        mergeable = sim >= self.tau
        return JudgeVerdict(
            mergeable=mergeable,
            relation=Relation.EXACT if mergeable else Relation.DISTINCT,
            similarity=sim,
        )


# --------------------------------------------------------------------------------------------
# LLM-on-descriptions — built behind a client protocol; runs ONLY with a real client.
# --------------------------------------------------------------------------------------------


class LlmUnavailableError(RuntimeError):
    """Raised when an LLM judge is asked to run without a usable client (no fabrication)."""


@runtime_checkable
class LlmClient(Protocol):
    """A minimal text-completion client (Ollama / Anthropic adapters slot in here)."""

    name: str

    def complete(self, prompt: str) -> str: ...


class UnavailableLlmClient:
    """The default client when no LLM is configured — every call fails loudly.

    The runner detects this and records the LLM track as *not run* rather than inventing
    answers (CLAUDE.md §5: "Not yet run" is acceptable; fabricated results are not).
    """

    name = "unavailable"

    def complete(self, prompt: str) -> str:
        raise LlmUnavailableError(
            "no LLM client configured (set ANTHROPIC_API_KEY or run Ollama); "
            "the LLM-on-descriptions track was not run"
        )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Drop ``<think>…</think>`` reasoning blocks (Qwen3 etc.) so parsing sees the answer."""
    return _THINK_RE.sub("", text).strip()


class OpenAIClient:
    """An OpenAI-compatible chat-completions client (dependency-free, stdlib HTTP).

    Talks to the OpenAI ``/v1/chat/completions`` endpoint that ``vllm serve`` exposes, so a
    locally-served model (e.g. Qwen3 via ``start_vlm.sh``) can drive the LLM-on-descriptions
    judge — or the curation agent — through the same :class:`LlmClient` protocol as
    :class:`~viscurate.curation.agent.OllamaClient`. No dependency on the ``openai`` package and
    no network at import time: the request happens only in :meth:`complete`.

    The API key (when an endpoint needs one) is read from ``OPENAI_API_KEY``; vLLM ignores it, so
    it stays optional and is never written in source (org policy / CLAUDE.md §5). Set
    ``enable_thinking=False`` for reasoning models (Qwen3) so the one-word relation answer is not
    buried in (or truncated by) a ``<think>`` block — vLLM forwards it to the chat template.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8001/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        enable_thinking: bool | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.name = f"openai:{model}"

    def complete(self, prompt: str) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        is_hosted_openai = "api.openai.com" in self.base_url
        if not is_hosted_openai:
            body["temperature"] = self.temperature
        token_field = "max_completion_tokens" if is_hosted_openai else "max_tokens"
        body[token_field] = self.max_tokens
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise LlmUnavailableError(
                f"OpenAI-compatible request failed ({self.base_url}): "
                f"HTTP {exc.code} {exc.reason}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LlmUnavailableError(
                f"OpenAI-compatible request failed ({self.base_url}): {exc}"
            ) from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmUnavailableError(f"unexpected chat-completions response: {data!r}") from exc
        return _strip_reasoning(str(content))


_RELATION_KEYWORDS: tuple[tuple[str, Relation], ...] = (
    ("EXACT", Relation.EXACT),
    ("PERCEPTUAL", Relation.PERCEPTUAL),
    ("SUBSUMPTION", Relation.SUBSUMPTION),
    ("SEMANTIC", Relation.SEMANTIC_PRESERVING),
    ("COMPLEMENTARY", Relation.COMPLEMENTARY),
    ("DISTINCT", Relation.DISTINCT),
)

_LLM_PROMPT = """\
You are judging whether two image-processing skills are equivalent, using ONLY their text \
descriptions (you cannot run them). Classify the pair into exactly one relation:
- EXACT: identical output for all inputs
- PERCEPTUAL: visually indistinguishable output
- SUBSUMPTION: one is a special case of the other
- SEMANTIC: same kind of transformation, different algorithm
- COMPLEMENTARY: orthogonal operations that compose
- DISTINCT: genuinely different operations

Skill A — {a_name}: {a_desc}
Skill B — {b_name}: {b_desc}

Answer with one word: the relation."""


@dataclass(frozen=True)
class LlmJudge:
    """An LLM judge over descriptions. ``client`` defaults to the unavailable sentinel."""

    client: LlmClient = field(default_factory=UnavailableLlmClient)
    name: str = field(default="llm-on-descriptions", init=False)

    @property
    def available(self) -> bool:
        return not isinstance(self.client, UnavailableLlmClient)

    def verdict(self, a: TextRecord, b: TextRecord) -> JudgeVerdict:
        reply = self.client.complete(
            _LLM_PROMPT.format(
                a_name=a.name, a_desc=a.description, b_name=b.name, b_desc=b.description
            )
        )
        relation = self._parse(reply)
        mergeable = relation in (Relation.EXACT, Relation.PERCEPTUAL)
        return JudgeVerdict(
            mergeable=mergeable, relation=relation, similarity=1.0 if mergeable else 0.0
        )

    @staticmethod
    def _parse(reply: str) -> Relation:
        upper = reply.upper()
        for keyword, relation in _RELATION_KEYWORDS:
            if keyword in upper:
                return relation
        return Relation.DISTINCT  # unparseable → conservative (no merge)

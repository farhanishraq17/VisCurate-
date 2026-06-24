"""Phase 4 — the text baselines (CLAUDE.md §3.4).

These judges read names/descriptions only; they are the strawmen the output verifier must
beat. Tests are deterministic (no ML, no network): the TF-IDF embedder is dependency-light and
the LLM judge is exercised with a fake client (and confirmed to refuse to fabricate without
one).
"""

from __future__ import annotations

import pytest

from viscurate.baselines.judges import (
    EmbeddingCosineJudge,
    LlmJudge,
    LlmUnavailableError,
    NameMatchJudge,
    TextRecord,
    TfidfEmbedder,
    UnavailableLlmClient,
    text_record_from_spec,
)
from viscurate.equivalence.relations import Relation
from viscurate.skills.model import SkillMetadata, SkillSpec


def _rec(rid: str, name: str, desc: str, tag: str = "fam") -> TextRecord:
    return TextRecord(id=rid, name=name, description=desc, tags=(tag,))


def test_name_match_merges_similar_names_not_different() -> None:
    judge = NameMatchJudge(tau=0.5)
    same = judge.verdict(
        _rec("gaussian_blur_v1", "Gaussian blur", "x"),
        _rec("gaussian_blur_v2", "Gaussian blur", "x"),
    )
    assert same.mergeable and same.relation is Relation.EXACT
    diff = judge.verdict(
        _rec("blur_gaussian_v1", "Gaussian blur", "x"),
        _rec("sepia_tone_v1", "Sepia tone", "y"),
    )
    assert not diff.mergeable and diff.relation is Relation.DISTINCT


def test_tfidf_embedder_is_deterministic_and_normalized() -> None:
    corpus = ["gaussian blur kernel", "box blur kernel", "sepia warm tone"]
    emb = TfidfEmbedder(corpus)
    v1 = emb.embed("gaussian blur kernel")
    v2 = emb.embed("gaussian blur kernel")
    assert (v1 == v2).all()  # deterministic
    assert v1.shape == v2.shape
    import numpy as np

    assert float(np.linalg.norm(v1)) == pytest.approx(1.0, abs=1e-5)


def test_embedding_cosine_merges_similar_descriptions() -> None:
    # The strawman failure: two genuinely-different blurs share most description tokens.
    corpus = [
        "Gaussian blur Convolve with a Gaussian kernel blur",
        "Box blur Convolve with a uniform box kernel blur",
        "Sepia tone Apply a warm brown tint color",
    ]
    emb = TfidfEmbedder(corpus)
    judge = EmbeddingCosineJudge(emb, tau=0.3)
    blur = judge.verdict(
        _rec("blur_gaussian_v1", "Gaussian blur", "Convolve with a Gaussian kernel.", "blur"),
        _rec("blur_box_v1", "Box blur", "Convolve with a uniform box kernel.", "blur"),
    )
    assert blur.mergeable  # text says "merge" — the bias the output verifier must override
    far = judge.verdict(
        _rec("blur_gaussian_v1", "Gaussian blur", "Convolve with a Gaussian kernel.", "blur"),
        _rec("sepia_tone_v1", "Sepia tone", "Apply a warm brown tint.", "color"),
    )
    assert not far.mergeable


def test_llm_judge_unavailable_by_default_does_not_fabricate() -> None:
    judge = LlmJudge()
    assert judge.available is False
    with pytest.raises(LlmUnavailableError):
        judge.verdict(_rec("a_v1", "A", "desc"), _rec("b_v1", "B", "desc"))
    assert UnavailableLlmClient().name == "unavailable"


def test_llm_judge_parses_a_fake_client_reply() -> None:
    class FakeClient:
        name = "fake-llm"

        def complete(self, prompt: str) -> str:
            return "I think these are DISTINCT operations."

    judge = LlmJudge(client=FakeClient())
    assert judge.available
    v = judge.verdict(_rec("a_v1", "A", "x"), _rec("b_v1", "B", "y"))
    assert v.relation is Relation.DISTINCT and not v.mergeable


def test_text_record_from_spec_reads_text_surface() -> None:
    spec = SkillSpec(
        id="x_v1", name="X", description="does x", metadata=SkillMetadata(family="color")
    )
    rec = text_record_from_spec(spec)
    assert rec.name == "X" and rec.description == "does x" and rec.tags == ("color",)
    assert "does x" in rec.text() and "color" in rec.text()

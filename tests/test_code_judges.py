"""Tests for the A2 code-reading baselines (rungs 4, 5, 7).

DESTINATION: ``tests/test_code_judges.py`` in the VisCurate repo.

These are deterministic and dependency-light — no torch, no network, no model downloads — in
keeping with the repo's existing test discipline (the ML-dependent paths are marked ``slow``).

The load-bearing test is ``test_structure_only_is_blind_to_default_drift``. It asserts the
documented asymmetry that justifies shipping **two** AST variants: the structure-only variant is
blind to corruption defect type (v) — parameter-default drift — *by construction*. Reporting
only that variant would rig the baseline to fail, which is the mirror image of the strawman
problem the baseline ladder exists to avoid.

Verified output when written (2026-08-18)::

    default-drift pair:   structure-only=1.000   semantics-preserving=0.875
    renamed-vars pair:    structure-only=1.000   semantics-preserving=1.000
    different-ops pair:   semantics-preserving=0.400
"""

from __future__ import annotations

import pytest

from viscurate.baselines.code_judges import (
    SEMANTICS_PRESERVING,
    STRUCTURE_ONLY,
    AstCloneJudge,
    CodeRecord,
    _ast_tokens,
    _strip_docstring,
)
from viscurate.equivalence.relations import Relation

# --- fixtures: minimal skill sources exercising each documented case ---------------------------

BLUR_K3 = "def blur(img, k=3):\n    return cv2.GaussianBlur(img, (k, k), sigmaX=1.0)\n"
BLUR_K9 = "def blur(img, k=9):\n    return cv2.GaussianBlur(img, (k, k), sigmaX=1.0)\n"
BLUR_DOC = (
    'def blur(img, k=3):\n    """Blur it."""\n'
    "    return cv2.GaussianBlur(img, (k, k), sigmaX=1.0)\n"
)
BLUR_TMP = (
    "def blur(img, k=3):\n    tmp = cv2.GaussianBlur(img, (k, k), sigmaX=1.0)\n    return tmp\n"
)
BLUR_RENAMED = (
    "def blur(pic, n=3):\n    out = cv2.GaussianBlur(pic, (n, n), sigmaX=1.0)\n    return out\n"
)
CANNY = "def edges(img):\n    return cv2.Canny(img, 100, 200)\n"


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def _record(source: str, rid: str = "s") -> CodeRecord:
    return CodeRecord(id=rid, source=source, source_no_doc=_strip_docstring(source))


# --- the load-bearing asymmetry ---------------------------------------------------------------


def test_structure_only_is_blind_to_default_drift() -> None:
    """Structure-only canonicalizes literals, so a changed default is invisible to it.

    This is *why* both variants must be reported: corruption defect type (v) is exactly a
    corrupted default parameter, and a baseline that cannot see it is not a fair competitor.
    """
    assert (
        _jaccard(_ast_tokens(BLUR_K3, STRUCTURE_ONLY), _ast_tokens(BLUR_K9, STRUCTURE_ONLY)) == 1.0
    )


def test_semantics_preserving_sees_default_drift() -> None:
    """Retaining literal values makes the same pair separable — the fair, strong baseline."""
    sim = _jaccard(
        _ast_tokens(BLUR_K3, SEMANTICS_PRESERVING), _ast_tokens(BLUR_K9, SEMANTICS_PRESERVING)
    )
    assert sim < 1.0


# --- general clone-detection behaviour ---------------------------------------------------------


def test_alpha_renaming_collides_in_both_variants() -> None:
    """Renamed locals are the classic type-2 clone; both variants must still match."""
    for variant in (STRUCTURE_ONLY, SEMANTICS_PRESERVING):
        assert _jaccard(_ast_tokens(BLUR_TMP, variant), _ast_tokens(BLUR_RENAMED, variant)) == 1.0


def test_different_operations_do_not_collide() -> None:
    sim = _jaccard(
        _ast_tokens(BLUR_TMP, SEMANTICS_PRESERVING), _ast_tokens(CANNY, SEMANTICS_PRESERVING)
    )
    assert sim < 0.5


def test_called_api_name_is_retained_in_both_variants() -> None:
    """``cv2.GaussianBlur`` vs ``cv2.Canny`` is structure, not a literal — never canonicalized.

    Erasing it would destroy the strongest signal a structural baseline has, handicapping it.
    """
    assert any("GaussianBlur" in t for t in _ast_tokens(BLUR_K3, STRUCTURE_ONLY))
    assert any("Canny" in t for t in _ast_tokens(CANNY, STRUCTURE_ONLY))


# --- docstring stripping (rule 1: rung 4 must not leak rung 3's signal) ------------------------


def test_docstring_does_not_affect_tokens() -> None:
    assert _ast_tokens(BLUR_K3, SEMANTICS_PRESERVING) == _ast_tokens(BLUR_DOC, SEMANTICS_PRESERVING)


def test_strip_docstring_removes_only_the_docstring() -> None:
    stripped = _strip_docstring(BLUR_DOC)
    assert "Blur it." not in stripped
    assert "GaussianBlur" in stripped


def test_strip_docstring_leaves_a_valid_body_when_docstring_is_the_only_statement() -> None:
    src = 'def noop():\n    """Only a docstring."""\n'
    stripped = _strip_docstring(src)
    assert "Only a docstring" not in stripped
    compile(stripped, "<test>", "exec")  # must remain syntactically valid (Pass inserted)


# --- judge-level behaviour ---------------------------------------------------------------------


def test_judge_reports_variant_in_its_name() -> None:
    """The run manifest must record *which* variant produced a number."""
    assert AstCloneJudge(variant=STRUCTURE_ONLY).name == "ast-clone[structure-only]"
    assert AstCloneJudge(variant=SEMANTICS_PRESERVING).name == "ast-clone[semantics-preserving]"


def test_judge_merges_identical_source() -> None:
    v = AstCloneJudge(tau=0.9).verdict(_record(BLUR_K3, "a"), _record(BLUR_K3, "b"))
    assert v.mergeable and v.relation is Relation.EXACT


def test_judge_separates_different_operations() -> None:
    v = AstCloneJudge(tau=0.9).verdict(_record(BLUR_TMP, "a"), _record(CANNY, "b"))
    assert not v.mergeable and v.relation is Relation.DISTINCT


def test_unparseable_source_abstains_rather_than_returning_distinct() -> None:
    """A parse failure is a *failure*, not evidence of non-equivalence.

    Mapping it to DISTINCT would fold the baseline's failure rate into a safety number — the
    same reporting flaw the shipped ``LlmJudge`` has for unparseable replies.
    """
    v = AstCloneJudge().verdict(_record("def broken(:\n", "a"), _record(BLUR_K3, "b"))
    assert v.relation is Relation.UNCERTAIN
    assert not v.mergeable


@pytest.mark.parametrize("variant", [STRUCTURE_ONLY, SEMANTICS_PRESERVING])
def test_tokenization_is_deterministic(variant: str) -> None:
    assert _ast_tokens(BLUR_TMP, variant) == _ast_tokens(BLUR_TMP, variant)

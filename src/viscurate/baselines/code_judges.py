"""A2 rungs 4, 5, 7 — baselines that read a skill's SOURCE CODE.

DESTINATION: ``src/viscurate/baselines/code_judges.py`` in the VisCurate repo.

WHY THESE EXIST
---------------
``sec/0_abstract.tex`` and ``sec/1_intro.tex:15`` both claim a skill's *"name, description, **or
source code**"* is a poor proxy for behaviour. The shipped baselines (`name-match`,
`embedding-cosine`, `llm-on-descriptions`) only test **name and description**. The source-code
half of the paper's own claim is asserted and never evaluated — exactly the kind of gap R3 has
the expertise to find.

This module closes it with three rungs:

    rung 4  CodeEmbeddingJudge   — neural code embedding (docstrings STRIPPED)
    rung 5  AstCloneJudge        — structural clone detection, TWO variants
    rung 7  LlmSourceJudge       — a frontier LLM reading the full implementation

Rung 7 is the strongest possible non-execution baseline. If it still fails on pairs that execute
differently, the argument is essentially closed.

TWO DESIGN RULES THAT ARE LOAD-BEARING (do not "simplify" these away)
---------------------------------------------------------------------
1. **Docstrings are stripped for rung 4.** Without stripping, a "code embedding" is partly a
   description embedding and the rung stops testing what it claims to test.
2. **Rung 5 ships TWO variants.** Canonicalizing literals to type placeholders makes the baseline
   blind to defect type (v) — parameter-default drift — *by construction*. Reporting only that
   variant would be rigging a baseline to fail, the mirror image of the strawman problem. The
   semantics-preserving variant (literals + defaults + called APIs retained) is the one the
   paper's claim must actually beat.

MODALITY BOUNDARY
-----------------
``baselines/`` is the one package permitted to read text (CLAUDE.md §1.2, enforced as a *package*
boundary). Source code is likewise a text surface, so it belongs here and nowhere else. Nothing
in ``equivalence/`` may import this module.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from viscurate.baselines.judges import (
    JudgeVerdict,
    LlmClient,
    LlmUnavailableError,
    UnavailableLlmClient,
)
from viscurate.equivalence.relations import Relation
from viscurate.skills.model import Skill

__all__ = [
    "AstCloneJudge",
    "AstVariant",
    "CodeEmbedder",
    "CodeEmbeddingJudge",
    "CodeRecord",
    "LlmSourceJudge",
    "code_record_from_skill",
]

NDArrayF = npt.NDArray[np.float32]


# ==============================================================================================
# The code surface
# ==============================================================================================


@dataclass(frozen=True)
class CodeRecord:
    """The *source* surface of one skill — the code analogue of ``TextRecord``.

    ``source`` is the raw implementation. ``source_no_doc`` has the docstring removed, which is
    what rung 4 must embed (see module docstring, rule 1).
    """

    id: str
    source: str
    source_no_doc: str


def _strip_docstring(source: str) -> str:
    """Return ``source`` with the function's docstring removed, preserving everything else."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def code_record_from_skill(skill: Skill) -> CodeRecord:
    """Extract a skill's implementation source via ``inspect``.

    Raises ``OSError`` if the source is unavailable (e.g. a C extension or a dynamically
    constructed callable). Callers should record such skills as *excluded* with the reason
    rather than substituting an empty string — a silently empty source would make two unrelated
    skills look identical.
    """
    source = textwrap.dedent(inspect.getsource(skill.fn))
    return CodeRecord(id=skill.id, source=source, source_no_doc=_strip_docstring(source))


# ==============================================================================================
# Rung 4 — code-embedding
# ==============================================================================================


@runtime_checkable
class CodeEmbedder(Protocol):
    """Maps source text to an L2-normalized vector. Mirrors ``TextEmbedder``."""

    name: str

    def embed(self, source: str) -> NDArrayF: ...


@dataclass
class TransformerCodeEmbedder:
    """A neural code embedder (UniXcoder / CodeT5+ / CodeBERT), lazily loaded.

    Verify the exact model id on the HF hub at implementation time; these are indicative.
    """

    model_id: str = "microsoft/unixcoder-base"
    device: str = "cpu"
    max_length: int = 512
    name: str = field(default="", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"code-embedding[{self.model_id.split('/')[-1]}]")
        self._tok: object | None = None
        self._model: object | None = None

    def _ensure(self) -> tuple[object, object]:
        if self._model is None:
            try:
                import torch  # noqa: F401
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "TransformerCodeEmbedder needs `pip install transformers`"
                ) from exc
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        assert self._tok is not None and self._model is not None
        return self._tok, self._model

    def embed(self, source: str) -> NDArrayF:
        import torch

        tok, model = self._ensure()
        batch = tok(  # type: ignore[operator]
            source,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            out = model(**batch)  # type: ignore[operator]
        # Mean-pool over the attention mask (standard for encoder code models).
        hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vec = torch.nn.functional.normalize(pooled, p=2, dim=-1)[0]
        return vec.cpu().numpy().astype(np.float32)

    def close(self) -> None:
        self._tok = None
        self._model = None


@dataclass(frozen=True)
class CodeEmbeddingJudge:
    """Rung 4 — cosine similarity over embeddings of the **docstring-stripped** source."""

    embedder: CodeEmbedder
    tau: float = 0.9
    name: str = field(default="code-embedding", init=False)

    def verdict(self, a: CodeRecord, b: CodeRecord) -> JudgeVerdict:
        va = self.embedder.embed(a.source_no_doc)
        vb = self.embedder.embed(b.source_no_doc)
        sim = float(np.clip(np.dot(va, vb), -1.0, 1.0))
        sim01 = (sim + 1.0) / 2.0
        mergeable = sim01 >= self.tau
        return JudgeVerdict(
            mergeable=mergeable,
            relation=Relation.EXACT if mergeable else Relation.DISTINCT,
            similarity=sim01,
        )


# ==============================================================================================
# Rung 5 — AST clone detection (two variants)
# ==============================================================================================

AstVariant = str  # "structure-only" | "semantics-preserving"

STRUCTURE_ONLY: AstVariant = "structure-only"
SEMANTICS_PRESERVING: AstVariant = "semantics-preserving"


class _AstNormalizer(ast.NodeVisitor):
    """Serializes an AST to a token sequence under one of the two normalization variants.

    Both variants α-rename local identifiers (so renamed-variable clones still collide) and drop
    docstrings. They differ in exactly one respect:

    * ``structure-only``        — every literal becomes a type placeholder (``<int>``, ``<str>``…)
    * ``semantics-preserving``  — literal VALUES, keyword-argument names, and called API names
                                  are retained, so a changed default parameter is visible

    The second variant is the fair, strong baseline. See the module docstring, rule 2.
    """

    def __init__(self, variant: AstVariant) -> None:
        self.variant = variant
        self.tokens: list[str] = []
        self._names: dict[str, str] = {}

    def _alpha(self, name: str) -> str:
        # Preserve dotted API roots (cv2, np, PIL) — renaming those would erase the strongest
        # semantic signal available to a structural baseline.
        if name in {"cv2", "np", "numpy", "PIL", "Image", "ImageOps", "ImageFilter", "skimage"}:
            return name
        if name not in self._names:
            self._names[name] = f"v{len(self._names)}"
        return self._names[name]

    def generic_visit(self, node: ast.AST) -> None:
        self.tokens.append(type(node).__name__)
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.tokens.append(f"Name:{self._alpha(node.id)}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Attribute access carries the called-API name — keep it in both variants; it is
        # structure, not a literal.
        self.tokens.append(f"Attr:{node.attr}")
        self.visit(node.value)

    def visit_Constant(self, node: ast.Constant) -> None:
        if self.variant == SEMANTICS_PRESERVING:
            self.tokens.append(f"Const:{node.value!r}")
        else:
            self.tokens.append(f"Const:<{type(node.value).__name__}>")

    def visit_keyword(self, node: ast.keyword) -> None:
        if self.variant == SEMANTICS_PRESERVING and node.arg:
            self.tokens.append(f"kw:{node.arg}")
        else:
            self.tokens.append("kw")
        self.visit(node.value)


def _ast_tokens(source: str, variant: AstVariant) -> list[str]:
    tree = ast.parse(_strip_docstring(textwrap.dedent(source)))
    norm = _AstNormalizer(variant)
    norm.visit(tree)
    return norm.tokens


@dataclass(frozen=True)
class AstCloneJudge:
    """Rung 5 — structural clone detection over normalized ASTs.

    Similarity is token-multiset Jaccard over the normalized serialization. This is cheaper than
    Zhang–Shasha tree edit distance and near-equivalent in practice for clone detection; swap in
    a tree-edit metric if the paper wants the stricter formulation.

    **Run both variants and report both.** See the module docstring, rule 2.
    """

    variant: AstVariant = SEMANTICS_PRESERVING
    tau: float = 0.9
    name: str = field(default="", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"ast-clone[{self.variant}]")

    def verdict(self, a: CodeRecord, b: CodeRecord) -> JudgeVerdict:
        try:
            ta = _ast_tokens(a.source, self.variant)
            tb = _ast_tokens(b.source, self.variant)
        except SyntaxError:
            # Unparseable source is a genuine failure, not a DISTINCT verdict. Record it as an
            # abstention so the failure rate stays visible (see LlmSourceJudge for the rationale).
            return JudgeVerdict(mergeable=False, relation=Relation.UNCERTAIN, similarity=0.0)

        sa, sb = set(ta), set(tb)
        union = sa | sb
        sim = len(sa & sb) / len(union) if union else 0.0
        mergeable = sim >= self.tau
        return JudgeVerdict(
            mergeable=mergeable,
            relation=Relation.EXACT if mergeable else Relation.DISTINCT,
            similarity=sim,
        )


# ==============================================================================================
# Rung 7 — LLM reading the full source
# ==============================================================================================

_LLM_SOURCE_PROMPT = """\
You are judging whether two image-processing skills are behaviourally equivalent, using ONLY \
their source code (you cannot run them). Classify the pair into exactly one relation:
- EXACT: identical output for all inputs
- PERCEPTUAL: visually indistinguishable output
- SUBSUMPTION: one is a special case of the other
- SEMANTIC: same kind of transformation, different algorithm
- COMPLEMENTARY: orthogonal operations that compose
- DISTINCT: genuinely different operations

Skill A source:
```python
{a_src}
```

Skill B source:
```python
{b_src}
```

Answer with one word: the relation."""


@dataclass(frozen=True)
class LlmSourceJudge:
    """Rung 7 — the strongest non-execution baseline: a frontier LLM reading the implementation.

    Two protocol decisions that differ **deliberately** from the shipped ``LlmJudge``:

    1. **An unparseable reply is ``UNCERTAIN``, not ``DISTINCT``.** The shipped judge maps
       unparseable output to DISTINCT "conservatively", which folds model failure into a safety
       number — the baseline scores *safer* precisely when it fails hardest, and its failure rate
       becomes invisible. Recording the abstention keeps coverage and risk separable.
       Report: safety among parsed answers, alongside the unparsed fraction.

    2. **Repeat the call to measure nondeterminism.** At temperature 0 there is no seed to vary,
       so "three seeds" is meaningless unless the API exposes real seed control. Call
       ``verdict`` ``repeats`` times and report dispersion; a single categorical draw from an LLM
       is not a measurement.

    The model MUST be disjoint from every curation subject, as ``sec/6_methodology.tex:163``
    already promises for the description judge.
    """

    client: LlmClient = field(default_factory=UnavailableLlmClient)
    repeats: int = 3
    name: str = field(default="llm-on-source", init=False)

    @property
    def available(self) -> bool:
        return not isinstance(self.client, UnavailableLlmClient)

    def verdict(self, a: CodeRecord, b: CodeRecord) -> JudgeVerdict:
        """Majority verdict over ``repeats`` identical calls (ties resolve to the safer label)."""
        votes: list[Relation] = []
        for _ in range(max(1, self.repeats)):
            reply = self.client.complete(_LLM_SOURCE_PROMPT.format(a_src=a.source, b_src=b.source))
            votes.append(self._parse(reply))

        # Majority; on a tie prefer the non-mergeable label (safety-first, and stated in the paper).
        counts = {r: votes.count(r) for r in set(votes)}
        best = max(counts.values())
        winners = [r for r, c in counts.items() if c == best]
        relation = (
            winners[0]
            if len(winners) == 1
            else next(
                (r for r in winners if r not in (Relation.EXACT, Relation.PERCEPTUAL)), winners[0]
            )
        )
        mergeable = relation in (Relation.EXACT, Relation.PERCEPTUAL)
        agreement = best / len(votes)
        return JudgeVerdict(mergeable=mergeable, relation=relation, similarity=agreement)

    @staticmethod
    def _parse(reply: str) -> Relation:
        upper = reply.upper()
        for keyword, relation in (
            ("EXACT", Relation.EXACT),
            ("PERCEPTUAL", Relation.PERCEPTUAL),
            ("SUBSUMPTION", Relation.SUBSUMPTION),
            ("SEMANTIC", Relation.SEMANTIC_PRESERVING),
            ("COMPLEMENTARY", Relation.COMPLEMENTARY),
            ("DISTINCT", Relation.DISTINCT),
        ):
            if keyword in upper:
                return relation
        # Unparseable → abstain. NOT DISTINCT. See the class docstring, decision 1.
        return Relation.UNCERTAIN


def llm_source_judge_or_none(client: LlmClient | None) -> LlmSourceJudge | None:
    """Return a judge only when a client is configured; else ``None`` so the runner records
    the track as *not run* rather than fabricating verdicts (CLAUDE.md §5)."""
    if client is None:
        return None
    try:
        client.complete("ping")
    except LlmUnavailableError:
        return None
    return LlmSourceJudge(client=client)

"""Query-stream artifacts for downstream evaluation (CLAUDE.md Phase 7).

The query layer is Layer E: natural-language tasks, query relevance, and held-out inputs used to
score whether a curated library still solves image tasks. A query manifest is an artifact, not a
result table: it records the input/reference hashes and the clean reference pipeline that created
the target output. Success numbers are produced only by :mod:`viscurate.downstream.evaluate`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from viscurate.skills.canonicalize import CANON_VERSION

__all__ = [
    "ParamValue",
    "PredicateKind",
    "PredicateSpec",
    "Query",
    "QueryManifest",
    "QuerySplit",
    "QueryStep",
    "load_query_manifest",
]

ParamValue: TypeAlias = bool | int | float | str | None
QuerySplit: TypeAlias = str


class PredicateKind(StrEnum):
    """Task-specific predicates layered on top of reference-output matching."""

    EXACT_SHAPE = "exact_shape"
    CHANNELS_EQUAL = "channels_equal"
    BINARY_MASK = "binary_mask"
    RGBA = "rgba"
    CHANGED_FROM_INPUT = "changed_from_input"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QueryStep(_Frozen):
    """One skill invocation in the clean reference pipeline."""

    skill_id: str
    params: dict[str, ParamValue] = Field(default_factory=dict)


class PredicateSpec(_Frozen):
    """One output predicate. ``height``/``width`` are used by ``exact_shape`` when supplied."""

    kind: PredicateKind
    height: int | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    tolerance: float = Field(default=1.0 / 255.0, ge=0)


class Query(_Frozen):
    """A single held-out downstream image task."""

    query_id: str
    split: QuerySplit
    instruction: str
    input_id: str
    input_sha256: str
    reference_sha256: str
    input_height: int = Field(gt=0)
    input_width: int = Field(gt=0)
    reference_height: int = Field(gt=0)
    reference_width: int = Field(gt=0)
    pipeline: tuple[QueryStep, ...]
    expected_skill_ids: tuple[str, ...]
    predicates: tuple[PredicateSpec, ...] = ()
    tags: tuple[str, ...] = ()

    @field_validator("input_sha256", "reference_sha256")
    @classmethod
    def _hex64(cls, v: str) -> str:
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v.lower()):
            raise ValueError("sha256 must be 64 hex chars")
        return v.lower()

    @model_validator(mode="after")
    def _pipeline_matches_expected(self) -> Query:
        step_ids = tuple(step.skill_id for step in self.pipeline)
        if not step_ids:
            raise ValueError(f"query {self.query_id!r} needs at least one reference step")
        if tuple(dict.fromkeys(step_ids)) != self.expected_skill_ids:
            raise ValueError(
                f"query {self.query_id!r}: expected_skill_ids must match pipeline order"
            )
        return self

    @property
    def input_path(self) -> Path:
        return Path("inputs") / f"{self.input_id}.npy"

    @property
    def reference_path(self) -> Path:
        return Path("references") / f"{self.query_id}.npy"


class QueryManifest(_Frozen):
    """The versioned query stream plus split and relevance helpers."""

    manifest_version: str = "1"
    generator_version: str
    canon_version: str = CANON_VERSION
    seed: int
    entries: tuple[Query, ...]

    @model_validator(mode="after")
    def _validate(self) -> QueryManifest:
        ids = [q.query_id for q in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("query_id values must be unique")
        inputs = [q.input_id for q in self.entries]
        if len(inputs) != len(set(inputs)):
            raise ValueError("input_id values must be unique")
        self.assert_split_disjoint_skills()
        return self

    def __len__(self) -> int:
        return len(self.entries)

    def splits(self) -> tuple[str, ...]:
        return tuple(sorted({q.split for q in self.entries}))

    def split_counts(self) -> dict[str, int]:
        out = {s: 0 for s in self.splits()}
        for q in self.entries:
            out[q.split] += 1
        return out

    def by_split(self, splits: Iterable[str] | None = None) -> tuple[Query, ...]:
        if splits is None:
            return self.entries
        wanted = set(splits)
        return tuple(q for q in self.entries if q.split in wanted)

    def referenced_skill_ids(self) -> frozenset[str]:
        return frozenset(sid for q in self.entries for sid in q.expected_skill_ids)

    def relevance(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for q in self.entries:
            for sid in q.expected_skill_ids:
                out.setdefault(sid, []).append(q.query_id)
        return {sid: tuple(qids) for sid, qids in sorted(out.items())}

    def assert_split_disjoint_skills(self) -> None:
        """Dev/test must be disjoint in skills, so solver tuning cannot leak through a skill."""
        by_split: dict[str, set[str]] = {}
        for q in self.entries:
            by_split.setdefault(q.split, set()).update(q.expected_skill_ids)
        splits = sorted(by_split)
        for i, a in enumerate(splits):
            for b in splits[i + 1 :]:
                overlap = by_split[a] & by_split[b]
                if overlap:
                    raise ValueError(f"query splits {a!r}/{b!r} share skill ids: {sorted(overlap)}")

    def assert_disjoint_from_probes(self, probe_hashes: Mapping[str, str] | Iterable[str]) -> None:
        """Raise if any query input reuses a probe image hash."""
        hashes = (
            set(probe_hashes.values()) if isinstance(probe_hashes, Mapping) else set(probe_hashes)
        )
        overlap = sorted({q.input_sha256 for q in self.entries} & hashes)
        if overlap:
            raise ValueError(f"query inputs overlap probe battery hashes: {overlap[:5]}")


def load_query_manifest(path: str | Path) -> QueryManifest:
    """Load either ``manifest.json`` inside a query dir or a direct manifest path."""
    p = Path(path)
    manifest_path = p / "manifest.json" if p.is_dir() else p
    return QueryManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

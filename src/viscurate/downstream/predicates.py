"""Task-specific predicates for downstream scoring (CLAUDE.md §3.3, Phase 7)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from viscurate.downstream.query import PredicateKind, PredicateSpec
from viscurate.skills.canonicalize import canonicalize, content_hash
from viscurate.skills.model import Image

__all__ = ["PredicateResult", "evaluate_predicate", "evaluate_predicates"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PredicateResult(_Frozen):
    """Result for one task predicate."""

    kind: PredicateKind
    passed: bool
    detail: str = ""


def _shape(img: Image) -> tuple[int, int]:
    return int(img.shape[0]), int(img.shape[1])


def _channels_equal(img: Image, tolerance: float) -> bool:
    if img.ndim == 2:
        return True
    if img.ndim != 3:
        return False
    if img.shape[2] == 1:
        return True
    if img.shape[2] < 3:
        return False
    arr = img[:, :, :3].astype(np.float32) / 255.0
    return bool(
        np.max(np.abs(arr[:, :, 0] - arr[:, :, 1])) <= tolerance
        and np.max(np.abs(arr[:, :, 1] - arr[:, :, 2])) <= tolerance
    )


def evaluate_predicate(
    spec: PredicateSpec, *, output: Image, reference: Image, input_image: Image
) -> PredicateResult:
    """Evaluate one task predicate on a solver output."""
    if spec.kind is PredicateKind.EXACT_SHAPE:
        expected = (
            int(spec.height) if spec.height is not None else int(reference.shape[0]),
            int(spec.width) if spec.width is not None else int(reference.shape[1]),
        )
        actual = _shape(output)
        return PredicateResult(
            kind=spec.kind,
            passed=actual == expected,
            detail=f"actual={actual}, expected={expected}",
        )

    if spec.kind is PredicateKind.CHANNELS_EQUAL:
        passed = _channels_equal(output, spec.tolerance)
        return PredicateResult(
            kind=spec.kind,
            passed=passed,
            detail=f"tolerance={spec.tolerance}",
        )

    if spec.kind is PredicateKind.BINARY_MASK:
        c = canonicalize(output)
        return PredicateResult(
            kind=spec.kind,
            passed=c.is_binary_mask,
            detail=f"shape={c.shape}, channels={c.n_input_channels}",
        )

    if spec.kind is PredicateKind.RGBA:
        passed = bool(output.ndim == 3 and output.shape[2] == 4)
        return PredicateResult(kind=spec.kind, passed=passed, detail=f"shape={tuple(output.shape)}")

    if spec.kind is PredicateKind.CHANGED_FROM_INPUT:
        out_hash = content_hash(canonicalize(output))
        in_hash = content_hash(canonicalize(input_image))
        return PredicateResult(kind=spec.kind, passed=out_hash != in_hash)

    raise ValueError(f"unknown predicate kind {spec.kind!r}")  # pragma: no cover


def evaluate_predicates(
    specs: Sequence[PredicateSpec], *, output: Image, reference: Image, input_image: Image
) -> tuple[PredicateResult, ...]:
    """Evaluate every task predicate in manifest order."""
    return tuple(
        evaluate_predicate(spec, output=output, reference=reference, input_image=input_image)
        for spec in specs
    )

"""Build the Phase-7 query stream and clean reference outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from viscurate.downstream.query import (
    PredicateKind,
    PredicateSpec,
    Query,
    QueryManifest,
    QueryStep,
)
from viscurate.probes.build import array_sha256
from viscurate.probes.manifest import ProbeManifest
from viscurate.rng import SeedManager
from viscurate.skills.canonicalize import CANON_VERSION
from viscurate.skills.model import Image
from viscurate.skills.registry import SkillRegistry

__all__ = ["DOWNSTREAM_GENERATOR_VERSION", "QueryBuildConfig", "build_query_stream", "load_array"]

DOWNSTREAM_GENERATOR_VERSION = "1.0.0"


class QueryBuildConfig(BaseModel):
    """Knobs for query-stream construction (``configs/queries.yaml``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 2026
    size: int = Field(default=96, gt=8)
    dev_repeats: int = Field(default=1, ge=1)
    test_repeats: int = Field(default=1, ge=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> QueryBuildConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)


@dataclass(frozen=True)
class _Template:
    split: str
    instruction: str
    steps: tuple[QueryStep, ...]
    predicates: tuple[PredicateSpec, ...]
    tags: tuple[str, ...]


_DEFAULT_TEMPLATES: tuple[_Template, ...] = (
    _Template(
        split="dev",
        instruction="Convert the image to BT.601 grayscale.",
        steps=(QueryStep(skill_id="grayscale_bt601_v1"),),
        predicates=(PredicateSpec(kind=PredicateKind.CHANNELS_EQUAL),),
        tags=("color", "grayscale"),
    ),
    _Template(
        split="dev",
        instruction="Resize the image to exactly 64 pixels wide and 48 pixels tall.",
        steps=(QueryStep(skill_id="resize_fixed_v1", params={"width": 64, "height": 48}),),
        predicates=(PredicateSpec(kind=PredicateKind.EXACT_SHAPE, height=48, width=64),),
        tags=("geometric", "resize"),
    ),
    _Template(
        split="dev",
        instruction="Create a binary luma mask using a threshold.",
        steps=(QueryStep(skill_id="threshold_binary_v1", params={"thresh": 128}),),
        predicates=(PredicateSpec(kind=PredicateKind.BINARY_MASK),),
        tags=("color", "mask"),
    ),
    _Template(
        split="dev",
        instruction="Rotate the image 90 degrees counter-clockwise.",
        steps=(QueryStep(skill_id="rotate_90_v1"),),
        predicates=(PredicateSpec(kind=PredicateKind.EXACT_SHAPE),),
        tags=("geometric", "rotate"),
    ),
    _Template(
        split="test",
        instruction="Invert the image into a photographic negative.",
        steps=(QueryStep(skill_id="invert_v1"),),
        predicates=(PredicateSpec(kind=PredicateKind.CHANGED_FROM_INPUT),),
        tags=("color", "invert"),
    ),
    _Template(
        split="test",
        instruction="Convert the image into an RGBA mask using luma as alpha.",
        steps=(QueryStep(skill_id="mask_to_rgba_v1", params={"thresh": 128}),),
        predicates=(PredicateSpec(kind=PredicateKind.RGBA),),
        tags=("mask", "rgba"),
    ),
    _Template(
        split="test",
        instruction="Apply a Gaussian blur to smooth the image.",
        steps=(QueryStep(skill_id="blur_gaussian_v1", params={"ksize": 5, "sigma": 0.0}),),
        predicates=(PredicateSpec(kind=PredicateKind.CHANGED_FROM_INPUT),),
        tags=("blur", "smooth"),
    ),
    _Template(
        split="test",
        instruction="Detect edges with the Canny edge detector.",
        steps=(QueryStep(skill_id="edges_canny_v1", params={"low": 80, "high": 180}),),
        predicates=(PredicateSpec(kind=PredicateKind.BINARY_MASK),),
        tags=("edges", "mask"),
    ),
)


def _write_array(path: Path, arr: Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr, allow_pickle=False)


def load_array(root: str | Path, rel_path: str | Path) -> Image:
    """Load a query input/reference array relative to a query directory."""
    arr: Image = np.load(Path(root) / rel_path, allow_pickle=False)
    return arr


def _query_input(sm: SeedManager, query_id: str, *, size: int, index: int) -> Image:
    """A deterministic held-out RGB image with enough structure for predicates and blur."""
    rng = sm.generator("query-input", query_id)
    h = size + (index % 2) * max(8, size // 6)
    w = size + ((index + 1) % 3) * max(8, size // 8)
    yy, xx = np.mgrid[:h, :w]
    base = np.empty((h, w, 3), dtype=np.uint8)
    base[:, :, 0] = (xx * 255 // max(1, w - 1)).astype(np.uint8)
    base[:, :, 1] = (yy * 255 // max(1, h - 1)).astype(np.uint8)
    base[:, :, 2] = rng.integers(20, 236, (h, w), dtype=np.uint8)
    for _ in range(4):
        y0 = int(rng.integers(0, max(1, h - 12)))
        x0 = int(rng.integers(0, max(1, w - 12)))
        y1 = min(h, y0 + int(rng.integers(8, max(9, h // 3))))
        x1 = min(w, x0 + int(rng.integers(8, max(9, w // 3))))
        color = rng.integers(0, 256, 3, dtype=np.uint8)
        base[y0:y1, x0:x1] = color
    return np.ascontiguousarray(base)


def _execute_reference(
    registry: SkillRegistry, image: Image, steps: tuple[QueryStep, ...], seed: int
) -> Image:
    out = image
    for step in steps:
        out = registry.get(step.skill_id).run(out, dict(step.params), seed=seed)
    return out


def _validate_templates(registry: SkillRegistry, templates: tuple[_Template, ...]) -> None:
    missing = sorted(
        {
            step.skill_id
            for template in templates
            for step in template.steps
            if step.skill_id not in registry
        }
    )
    if missing:
        raise ValueError(f"query templates reference unknown skill ids: {missing}")


def build_query_stream(
    cfg: QueryBuildConfig,
    registry: SkillRegistry,
    out_dir: str | Path,
    *,
    probe_manifest: ProbeManifest | None = None,
) -> QueryManifest:
    """Build query inputs + clean references and return the manifest."""
    _validate_templates(registry, _DEFAULT_TEMPLATES)
    out = Path(out_dir)
    inputs_dir = out / "inputs"
    refs_dir = out / "references"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    sm = SeedManager(cfg.seed)

    entries: list[Query] = []
    index = 0
    for template_i, template in enumerate(_DEFAULT_TEMPLATES):
        repeats = cfg.dev_repeats if template.split == "dev" else cfg.test_repeats
        for repeat in range(repeats):
            skill_slug = "-".join(step.skill_id.replace("_v1", "") for step in template.steps)
            query_id = f"{template.split}_{template_i:02d}_{repeat:02d}_{skill_slug}"
            input_id = f"{query_id}_input"
            image = _query_input(sm, query_id, size=cfg.size, index=index)
            reference = _execute_reference(
                registry,
                image,
                template.steps,
                seed=sm.child_seed("reference", query_id) & 0xFFFF_FFFF,
            )
            _write_array(inputs_dir / f"{input_id}.npy", image)
            _write_array(refs_dir / f"{query_id}.npy", reference)
            expected_skill_ids = tuple(dict.fromkeys(step.skill_id for step in template.steps))
            entries.append(
                Query(
                    query_id=query_id,
                    split=template.split,
                    instruction=template.instruction,
                    input_id=input_id,
                    input_sha256=array_sha256(image),
                    reference_sha256=array_sha256(reference),
                    input_height=int(image.shape[0]),
                    input_width=int(image.shape[1]),
                    reference_height=int(reference.shape[0]),
                    reference_width=int(reference.shape[1]),
                    pipeline=template.steps,
                    expected_skill_ids=expected_skill_ids,
                    predicates=template.predicates,
                    tags=template.tags,
                )
            )
            index += 1

    manifest = QueryManifest(
        generator_version=DOWNSTREAM_GENERATOR_VERSION,
        canon_version=CANON_VERSION,
        seed=cfg.seed,
        entries=tuple(entries),
    )
    if probe_manifest is not None:
        manifest.assert_disjoint_from_probes({e.probe_id: e.sha256 for e in probe_manifest.entries})
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest

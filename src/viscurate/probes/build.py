"""Probe-battery orchestrator (CLAUDE.md Phase 2 / §2.1).

Combines the synthetic backbone with the license-clean COCO natural photos, writes each probe
as a ``.npy`` array (dtype + channels preserved exactly), and emits a coverage-checked
:class:`ProbeManifest`. Reproducible: same seed → same probe bytes → same hashes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from viscurate.logging import get_logger
from viscurate.probes.coco import select_natural_probes
from viscurate.probes.manifest import CC0, ChannelFormat, ProbeEntry, ProbeManifest
from viscurate.probes.synthetics import generate_synthetic_probes
from viscurate.rng import SeedManager
from viscurate.skills.canonicalize import CANON_VERSION
from viscurate.skills.model import Image as Array

__all__ = ["PROBE_GENERATOR_VERSION", "ProbesConfig", "array_sha256", "build_battery", "load_probe"]

PROBE_GENERATOR_VERSION = "1.0.0"

_DEFAULT_COUNTS: dict[str, int] = {
    "gradient": 25,
    "texture": 30,
    "shape": 25,
    "document": 25,
    "colorchart": 10,
    "noise": 15,
}
_DEFAULT_DEGENERATE = (
    "all_black",
    "all_white",
    "single_color",
    "one_by_one",
    "thin_1xN",
    "thin_Nx1",
    "high_res_1024",
)


class ProbesConfig(BaseModel):
    """Knobs for the battery (configs/probes.yaml). Frozen + reject unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 1234
    size: int = 128
    counts: dict[str, int] = dict(_DEFAULT_COUNTS)
    n_natural: int = 40
    download_natural: bool = True
    min_per_domain: int = 5
    required_degenerate: tuple[str, ...] = _DEFAULT_DEGENERATE

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProbesConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)


def array_sha256(arr: Array) -> str:
    """Representation-independent content hash of a probe array (dtype + shape + bytes)."""
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _write_array(out_dir: Path, probe_id: str, arr: Array) -> None:
    np.save(out_dir / f"{probe_id}.npy", arr, allow_pickle=False)


def load_probe(out_dir: str | Path, probe_id: str) -> Array:
    arr: Array = np.load(Path(out_dir) / f"{probe_id}.npy", allow_pickle=False)
    return arr


def _hw(arr: Array) -> tuple[int, int]:
    return int(arr.shape[0]), int(arr.shape[1])


def build_battery(
    cfg: ProbesConfig,
    out_dir: str | Path,
    cache_dir: str | Path,
    *,
    timeout: float = 30.0,
) -> ProbeManifest:
    """Build the battery to ``out_dir`` and return the (coverage-checked) manifest."""
    log = get_logger("probes.build")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sm = SeedManager(cfg.seed)

    entries: list[ProbeEntry] = []

    for gp in generate_synthetic_probes(sm, cfg.counts, size=cfg.size):
        _write_array(out, gp.base_id, gp.array)
        h, w = _hw(gp.array)
        entries.append(
            ProbeEntry(
                probe_id=gp.base_id,
                sha256=array_sha256(gp.array),
                domain=gp.domain,
                channel_format=gp.channel_format,
                height=h,
                width=w,
                source="synthetic",
                license=CC0,
                attribution="self-generated",
                notes=gp.notes,
            )
        )

    if cfg.download_natural and cfg.n_natural > 0:
        try:
            natural = select_natural_probes(sm, cfg.n_natural, Path(cache_dir), timeout=timeout)
        except Exception as exc:
            log.warning("coco_fetch_failed", error=repr(exc))
            natural = []
        for fp in natural:
            _write_array(out, fp.base_id, fp.array)
            h, w = _hw(fp.array)
            entries.append(
                ProbeEntry(
                    probe_id=fp.base_id,
                    sha256=array_sha256(fp.array),
                    domain="natural",
                    channel_format="rgb",
                    height=h,
                    width=w,
                    source=fp.source,
                    license=fp.license,
                    attribution=fp.attribution,
                    notes=fp.notes,
                )
            )

    manifest = ProbeManifest(
        generator_version=PROBE_GENERATOR_VERSION,
        canon_version=CANON_VERSION,
        seed=cfg.seed,
        entries=tuple(entries),
    )
    required_formats: tuple[ChannelFormat, ...] = ("rgb", "rgba", "gray", "gray16", "palette")
    # The synthetic domains must each meet the floor; degenerate must cover the named cases;
    # natural is required only when the (optional) COCO fetch actually produced photos.
    required_domains: dict[str, int] = {d: cfg.min_per_domain for d in cfg.counts}
    required_domains["degenerate"] = 1
    if any(e.domain == "natural" for e in entries):
        required_domains["natural"] = 1
    manifest.assert_coverage(
        required_domains=required_domains,
        required_formats=required_formats,
        required_degenerate=cfg.required_degenerate,
    )
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    log.info("battery_built", n=len(entries), out_dir=str(out))
    return manifest

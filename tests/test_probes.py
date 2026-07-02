from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from viscurate.equivalence.param_alignment import load_param_alignment
from viscurate.probes.build import ProbesConfig, array_sha256, build_battery, load_probe
from viscurate.probes.coco import PERMISSIVE_LICENSE_IDS, coco_license_table, filter_permissive
from viscurate.probes.manifest import CC0, License, ProbeManifest
from viscurate.probes.oracle import freeze_oracle, freeze_sweep_oracle, verify_oracle
from viscurate.probes.synthetics import generate_synthetic_probes
from viscurate.rng import SeedManager
from viscurate.skills.library import build_builtin_registry

_SMALL_COUNTS = {
    "gradient": 5,
    "texture": 5,
    "shape": 5,
    "document": 5,
    "colorchart": 5,
    "noise": 5,
}


def _small_cfg() -> ProbesConfig:
    return ProbesConfig(
        seed=1234, size=32, counts=_SMALL_COUNTS, download_natural=False, min_per_domain=5
    )


# --- license / manifest ----------------------------------------------------------------


def test_license_rejects_unknown_and_empty() -> None:
    for bad in ("", "  ", "unknown", "Unknown"):
        with pytest.raises(ValidationError):
            License(name=bad)


def test_license_must_be_redistributable_with_derivatives() -> None:
    with pytest.raises(ValidationError):
        License(name="CC BY-NC", redistributable=False)
    with pytest.raises(ValidationError):
        License(name="CC BY-ND", allows_derivatives=False)
    assert CC0.allows_derivatives and CC0.redistributable


def test_probe_entry_validates_sha256(tmp_path) -> None:
    cfg = _small_cfg()
    manifest = build_battery(cfg, tmp_path / "p", tmp_path / "cache")
    assert all(len(e.sha256) == 64 for e in manifest.entries)


def test_coverage_failure_is_detected() -> None:
    # An empty manifest cannot satisfy any required domain.
    m = ProbeManifest(generator_version="1", canon_version="1.0.0", seed=1, entries=())
    with pytest.raises(ValueError):
        m.assert_coverage(required_domains={"gradient": 1})


# --- synthetics ------------------------------------------------------------------------


def test_synthetics_deterministic_and_cover_formats() -> None:
    a = generate_synthetic_probes(SeedManager(7), _SMALL_COUNTS, size=32)
    b = generate_synthetic_probes(SeedManager(7), _SMALL_COUNTS, size=32)
    assert [p.base_id for p in a] == [p.base_id for p in b]
    assert all(np.array_equal(x.array, y.array) for x, y in zip(a, b, strict=True))
    fmts = {p.channel_format for p in a}
    assert {"rgb", "rgba", "gray", "gray16", "palette"} <= fmts
    notes = {p.notes for p in a if p.domain == "degenerate"}
    assert {"all_black", "all_white", "one_by_one", "high_res_1024"} <= notes


def test_synthetics_dtypes() -> None:
    probes = generate_synthetic_probes(SeedManager(1), _SMALL_COUNTS, size=32)
    by_fmt = {p.channel_format: p for p in probes}
    assert by_fmt["gray16"].array.dtype == np.uint16
    assert by_fmt["rgba"].array.shape[2] == 4


# --- coco (pure filter logic, no network) ----------------------------------------------


def test_permissive_ids_and_license_table() -> None:
    assert PERMISSIVE_LICENSE_IDS == (4, 7, 8)
    table = coco_license_table()
    assert set(table) == {4, 7, 8}
    assert all(lic.redistributable and lic.allows_derivatives for lic in table.values())
    assert table[4].spdx == "CC-BY-2.0"


def test_filter_permissive_excludes_nc_and_nd() -> None:
    images = [
        {"id": 1, "license": 4},  # CC BY -> keep
        {"id": 2, "license": 6},  # NoDerivs -> drop (we make derivatives)
        {"id": 3, "license": 2},  # NonCommercial -> drop
        {"id": 4, "license": 7},  # no known restrictions -> keep
        {"id": 5, "license": 5},  # ShareAlike -> drop (not in {4,7,8})
    ]
    kept = {im["id"] for im in filter_permissive(images, PERMISSIVE_LICENSE_IDS)}
    assert kept == {1, 4}


# --- build orchestrator ----------------------------------------------------------------


def test_build_is_reproducible_and_covered(tmp_path) -> None:
    cfg = _small_cfg()
    m1 = build_battery(cfg, tmp_path / "a", tmp_path / "cache")
    m2 = build_battery(cfg, tmp_path / "b", tmp_path / "cache")
    h1 = {e.probe_id: e.sha256 for e in m1.entries}
    h2 = {e.probe_id: e.sha256 for e in m2.entries}
    assert h1 == h2  # same seed -> same hashes
    counts = m1.domain_counts()
    assert all(counts[d] >= 5 for d in _SMALL_COUNTS)
    assert (tmp_path / "a" / "manifest.json").exists()


def test_stored_probe_matches_recorded_hash(tmp_path) -> None:
    m = build_battery(_small_cfg(), tmp_path / "a", tmp_path / "cache")
    e = m.entries[0]
    assert array_sha256(load_probe(tmp_path / "a", e.probe_id)) == e.sha256


def test_manifest_round_trips(tmp_path) -> None:
    m = build_battery(_small_cfg(), tmp_path / "a", tmp_path / "cache")
    text = m.model_dump_json()
    reloaded = ProbeManifest.model_validate_json(text)
    assert len(reloaded) == len(m)
    assert reloaded.entries[0].license.name == m.entries[0].license.name


# --- oracle ----------------------------------------------------------------------------


def test_oracle_freeze_and_verify_clean(tmp_path) -> None:
    m = build_battery(_small_cfg(), tmp_path / "a", tmp_path / "cache")
    reg = build_builtin_registry()
    subset = [
        "flip_horizontal_v1",
        "blur_gaussian_v1",
        "grayscale_bt601_v1",
        "threshold_otsu_v1",
        "palette_reduce_kmeans_v1",
        "add_gaussian_noise_v1",
        "inpaint_telea_v1",
        "mask_to_rgba_v1",
    ]
    oracle = freeze_oracle(m, tmp_path / "a", reg, oracle_seed=0, skill_ids=subset)
    assert len(oracle.entries) == len(subset) * len(m)
    # The oracle is a stable reference: re-running reproduces every frozen pair.
    assert verify_oracle(oracle, m, tmp_path / "a", reg) == []


def test_oracle_seeded_skill_is_stable(tmp_path) -> None:
    m = build_battery(_small_cfg(), tmp_path / "a", tmp_path / "cache")
    reg = build_builtin_registry()
    o1 = freeze_oracle(m, tmp_path / "a", reg, oracle_seed=3, skill_ids=["add_gaussian_noise_v1"])
    o2 = freeze_oracle(m, tmp_path / "a", reg, oracle_seed=3, skill_ids=["add_gaussian_noise_v1"])
    h1 = {(e.skill_id, e.probe_id): e.output_sha256 for e in o1.entries}
    h2 = {(e.skill_id, e.probe_id): e.output_sha256 for e in o2.entries}
    assert h1 == h2  # fixed oracle_seed -> identical seeded-skill outputs


def test_sweep_oracle_freezes_parameterized_bindings(tmp_path) -> None:
    m = build_battery(_small_cfg(), tmp_path / "a", tmp_path / "cache")
    reg = build_builtin_registry()
    align = load_param_alignment("configs/param_alignment.yaml")
    oracle = freeze_sweep_oracle(
        m,
        tmp_path / "a",
        reg,
        align,
        oracle_seed=0,
        skill_ids=["blur_gaussian_v1", "crop_bounding_box_v1"],
    )
    assert oracle.artifact_kind == "sweep_oracle"
    assert oracle.alignment_version == align.version
    assert any(e.skill_id == "blur_gaussian_v1" and e.params_key != "{}" for e in oracle.entries)
    assert verify_oracle(oracle, m, tmp_path / "a", reg) == []

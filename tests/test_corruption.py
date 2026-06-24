"""Phase 5 — corruption generator tests (CLAUDE.md Phase-5 exit criteria).

Exercises the four exit criteria explicitly:

1. **same seed → byte-identical library** (plan + apply are deterministic / pure);
2. **expected per-type counts** (eligibility-aware Hamilton apportionment, honest realized counts);
3. **buggy skills measurably wrong vs the clean reference** (per-type QA assertions);
4. **G_ρ derived deterministically from G0 + log** (deltas appended, never relabeled; DAG/closure).

All deterministic and ML-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from viscurate.corruption import (
    apply_corruption,
    composition_by_name,
    generate_instance,
    load_g0_spec,
    plan_corruption,
    run_qa,
)
from viscurate.corruption.grid import CorruptionGridConfig
from viscurate.corruption.types import Composition, CorruptionType, IdealActionKind
from viscurate.equivalence.relations import Relation
from viscurate.skills.library import build_builtin_registry
from viscurate.skills.model import Image

G0_PATH = "configs/ground_truth_g0.yaml"


@pytest.fixture(scope="module")
def l0():
    return build_builtin_registry().all()


@pytest.fixture(scope="module")
def g0():
    return load_g0_spec(G0_PATH)


@pytest.fixture(scope="module")
def battery() -> list[tuple[str, Image]]:
    """A small multi-domain battery covering every domain a defect can target."""
    rng = np.random.default_rng(7)
    checker = (np.indices((48, 64)).sum(0) % 2 * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (48, 1))[:, :, None].repeat(3, 2)
    return [
        ("rgb", rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)),
        ("rgba", rng.integers(0, 256, (48, 64, 4), dtype=np.uint8)),
        ("gray", rng.integers(0, 256, (48, 64), dtype=np.uint8)),
        ("u16", rng.integers(0, 65536, (48, 64, 3), dtype=np.uint16)),
        ("grad", grad),
        ("checker", checker),
        ("white", np.full((48, 64, 3), 255, np.uint8)),
    ]


def only(t: CorruptionType) -> Composition:
    """A composition putting all weight on one defect type."""
    return Composition(name=f"only_{t.value}", weights={t: 1.0})


# --------------------------------------------------------------------------------------
# Exit criterion 1 — determinism / purity.
# --------------------------------------------------------------------------------------


def test_plan_is_deterministic(l0):
    comp = composition_by_name("uniform")
    a = plan_corruption(l0, rho=0.3, composition=comp, seed=1234, mode="single")
    b = plan_corruption(l0, rho=0.3, composition=comp, seed=1234, mode="single")
    assert a.model_dump_json() == b.model_dump_json()


def test_apply_is_pure_byte_identical(l0, g0):
    comp = composition_by_name("duplicate_heavy")
    log = plan_corruption(l0, rho=0.5, composition=comp, seed=99, mode="single")
    lib1 = apply_corruption(l0, log, g0)
    lib2 = apply_corruption(l0, log, g0)
    assert lib1.registry.to_json() == lib2.registry.to_json()
    assert lib1.g_rho_spec.model_dump_json() == lib2.g_rho_spec.model_dump_json()
    assert [a.model_dump() for a in lib1.ideal_actions] == [
        a.model_dump() for a in lib2.ideal_actions
    ]


def test_same_seed_identical_library_across_full_pipeline(l0, g0):
    comp = composition_by_name("metadata_heavy")
    libs = [
        apply_corruption(
            l0, plan_corruption(l0, rho=0.4, composition=comp, seed=7, mode="mixed"), g0
        )
        for _ in range(2)
    ]
    assert libs[0].registry.to_json() == libs[1].registry.to_json()


def test_different_seed_changes_sites_but_not_counts(l0):
    comp = composition_by_name("uniform")
    a = plan_corruption(l0, rho=0.3, composition=comp, seed=1, mode="single")
    b = plan_corruption(l0, rho=0.3, composition=comp, seed=2, mode="single")
    assert a.realized_counts() == b.realized_counts()  # counts are seed-independent
    assert a.sites() != b.sites()  # site selection is seeded


# --------------------------------------------------------------------------------------
# Exit criterion 2 — ρ semantics and expected per-type counts.
# --------------------------------------------------------------------------------------


def test_rho_sets_site_count(l0):
    comp = composition_by_name("uniform")
    for rho, k in [(0.1, 10), (0.3, 30), (0.5, 50), (1.0, 100)]:
        log = plan_corruption(l0, rho=rho, composition=comp, seed=5, mode="single")
        assert len(log.sites()) == k
        assert len(log.entries) == k  # single-defect: one entry per site


def test_uniform_counts_are_hamilton(l0):
    log = plan_corruption(
        l0, rho=0.3, composition=composition_by_name("uniform"), seed=5, mode="single"
    )
    # 30 over 7 types: floor 4 each (28) + 2 to the first two enum types.
    counts = log.realized_counts()
    assert sum(counts.values()) == 30
    assert counts["implementation_bug"] == 5
    assert counts["metadata_mislead"] == 5
    assert all(
        counts[t.value] == 4
        for t in CorruptionType
        if t.value not in {"implementation_bug", "metadata_mislead"}
    )


def test_single_type_composition_concentrates(l0):
    log = plan_corruption(
        l0, rho=0.1, composition=only(CorruptionType.DUPLICATE), seed=3, mode="single"
    )
    counts = log.realized_counts()
    assert counts["duplicate"] == 10
    assert sum(v for k, v in counts.items() if k != "duplicate") == 0


def test_restricted_pool_deficit_spills_and_is_recorded(l0):
    # All weight on SUBSUMPTION at ρ=1.0: the numeric-param pool cannot fill 100 sites, so the
    # deficit must spill to flexible types and the realized counts must stay honest (sum == 100).
    log = plan_corruption(
        l0, rho=1.0, composition=only(CorruptionType.SUBSUMPTION), seed=3, mode="single"
    )
    counts = log.realized_counts()
    assert sum(counts.values()) == 100
    assert counts["subsumption"] < 100  # capped by the eligible pool
    assert counts["implementation_bug"] > 0  # spill landed on the fallback flexible type


# --------------------------------------------------------------------------------------
# Exit criterion 3 — QA: defects took effect (the asymmetry that splits verifier/agent).
# --------------------------------------------------------------------------------------


def test_qa_all_pass_uniform_single(l0, g0, battery):
    lib = apply_corruption(
        l0,
        plan_corruption(
            l0, rho=0.5, composition=composition_by_name("uniform"), seed=11, mode="single"
        ),
        g0,
    )
    qa = run_qa(l0, lib, battery)
    assert qa.all_passed, [r.reason for r in qa.results if r.status == "fail"]


@pytest.mark.parametrize("comp", ["uniform", "duplicate_heavy", "metadata_heavy"])
@pytest.mark.parametrize("mode", ["single", "mixed"])
def test_qa_all_pass_full_grid_point(l0, g0, battery, comp, mode):
    lib = apply_corruption(
        l0,
        plan_corruption(l0, rho=1.0, composition=composition_by_name(comp), seed=2024, mode=mode),
        g0,
    )
    qa = run_qa(l0, lib, battery)
    fails = [(r.type.value, r.site_id, r.reason) for r in qa.results if r.status == "fail"]
    assert not fails, fails


def test_implementation_bug_diverges_and_is_flagged(l0, g0, battery):
    lib = apply_corruption(
        l0,
        plan_corruption(
            l0, rho=0.1, composition=only(CorruptionType.IMPLEMENTATION_BUG), seed=4, mode="single"
        ),
        g0,
    )
    qa = run_qa(l0, lib, battery)
    assert qa.all_passed
    for e in lib.log.entries:
        assert lib.registry.get(e.site_id).metadata.is_buggy is True


def test_metadata_mislead_leaves_outputs_unchanged(l0, g0, battery):
    lib = apply_corruption(
        l0,
        plan_corruption(
            l0, rho=0.1, composition=only(CorruptionType.METADATA_MISLEAD), seed=4, mode="single"
        ),
        g0,
    )
    clean = {s.id: s for s in l0}
    qa = run_qa(l0, lib, battery)
    assert qa.all_passed
    for e in lib.log.entries:
        corrupt = lib.registry.get(e.site_id)
        assert corrupt.description != clean[e.site_id].description  # text changed
        # output unchanged on an RGB probe (fn untouched)
        _pid, img = battery[0]
        np.testing.assert_array_equal(corrupt.run(img, seed=0), clean[e.site_id].run(img, seed=0))


def test_param_schema_bug_changes_default_keeps_fn(l0, g0, battery):
    lib = apply_corruption(
        l0,
        plan_corruption(
            l0, rho=0.1, composition=only(CorruptionType.PARAM_SCHEMA_BUG), seed=4, mode="single"
        ),
        g0,
    )
    clean = {s.id: s for s in l0}
    qa = run_qa(l0, lib, battery)
    assert qa.all_passed
    for e in lib.log.entries:
        corrupt = lib.registry.get(e.site_id)
        base = clean[e.site_id]
        assert corrupt.params_schema.defaults() != base.params_schema.defaults()
        # fn identical at matched (clean) params
        _pid, img = battery[0]
        same_params = base.params_schema.defaults()
        np.testing.assert_array_equal(
            corrupt.run(img, same_params, 0), base.run(img, same_params, 0)
        )


def test_domain_scoped_bug_is_conditional(l0, g0):
    lib = apply_corruption(
        l0,
        plan_corruption(
            l0, rho=0.05, composition=only(CorruptionType.DOMAIN_SCOPED_BUG), seed=4, mode="single"
        ),
        g0,
    )
    clean = {s.id: s for s in l0}
    rgb = np.random.default_rng(1).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    for e in lib.log.entries:
        corrupt, base = lib.registry.get(e.site_id), clean[e.site_id]
        # unchanged on RGB (the bug is scoped to a non-RGB domain)
        np.testing.assert_array_equal(corrupt.run(rgb, seed=0), base.run(rgb, seed=0))


# --------------------------------------------------------------------------------------
# Exit criterion 4 — G_ρ derived from G0 + log; ideal-action key.
# --------------------------------------------------------------------------------------


def test_g_rho_keeps_planted_g0_relations(l0, g0):
    lib = apply_corruption(
        l0,
        plan_corruption(
            l0, rho=0.3, composition=composition_by_name("uniform"), seed=5, mode="single"
        ),
        g0,
    )
    # A planted G0 subsumption and hard-negative survive untouched.
    assert (
        lib.g_rho.label("rotate_90_v1", "rotate_canvas_degrees_v1").relation is Relation.SUBSUMPTION
    )
    assert lib.g_rho.label("blur_gaussian_v1", "blur_box_v1").relation is Relation.DISTINCT
    assert lib.g_rho.is_hard_negative("blur_gaussian_v1", "blur_box_v1")


def test_g_rho_injects_duplicate_and_subsumption_relations(l0, g0):
    log = plan_corruption(
        l0, rho=0.2, composition=composition_by_name("duplicate_heavy"), seed=8, mode="single"
    )
    lib = apply_corruption(l0, log, g0)
    for e in log.entries:
        if e.type is CorruptionType.DUPLICATE:
            rel = lib.g_rho.label(e.site_id, e.new_skill_id).relation
            expected = Relation.PERCEPTUAL if e.variant == "perceptual" else Relation.EXACT
            assert rel is expected
        elif e.type is CorruptionType.SUBSUMPTION:
            assert lib.g_rho.label(e.new_skill_id, e.site_id).relation is Relation.SUBSUMPTION


def test_dead_skill_has_no_injected_relation_and_is_flagged(l0, g0):
    log = plan_corruption(
        l0, rho=0.1, composition=only(CorruptionType.DEAD_SKILL), seed=8, mode="single"
    )
    lib = apply_corruption(l0, log, g0)
    for e in log.entries:
        dead = lib.registry.get(e.new_skill_id)
        assert dead.metadata.is_dead is True
        # the dead skill participates in no injected relation: DISTINCT from its donor
        assert lib.g_rho.label(e.new_skill_id, e.site_id).relation is Relation.DISTINCT


def test_ideal_action_key_matches_types(l0, g0):
    log = plan_corruption(
        l0, rho=0.5, composition=composition_by_name("uniform"), seed=12, mode="single"
    )
    lib = apply_corruption(l0, log, g0)
    expected = {
        CorruptionType.IMPLEMENTATION_BUG: IdealActionKind.REMOVE,
        CorruptionType.METADATA_MISLEAD: IdealActionKind.MODIFY,
        CorruptionType.PARAM_SCHEMA_BUG: IdealActionKind.MODIFY,
        CorruptionType.DOMAIN_SCOPED_BUG: IdealActionKind.MODIFY,
        CorruptionType.DUPLICATE: IdealActionKind.MERGE,
        CorruptionType.SUBSUMPTION: IdealActionKind.PARAMETERIZE,
        CorruptionType.DEAD_SKILL: IdealActionKind.REMOVE,
    }
    assert len(lib.ideal_actions) == len(log.entries)
    for a in lib.ideal_actions:
        assert a.kind is expected[a.type]


def test_g_rho_validates_dag_and_is_constructible_at_rho_1(l0, g0):
    # duplicate_heavy at ρ=1.0 injects many EXACT/PERCEPTUAL + subsumption edges; constructing
    # G_ρ runs the DAG + EXACT-closure validators — a raise here is a real failure.
    log = plan_corruption(
        l0, rho=1.0, composition=composition_by_name("duplicate_heavy"), seed=33, mode="single"
    )
    lib = apply_corruption(l0, log, g0)
    assert len(lib.registry) == len(l0) + lib.n_added()


# --------------------------------------------------------------------------------------
# Library integrity + grid driver + input validation.
# --------------------------------------------------------------------------------------


def test_corrupted_registry_has_unique_ids_and_serializes(l0, g0):
    lib = apply_corruption(
        l0,
        plan_corruption(
            l0, rho=0.6, composition=composition_by_name("uniform"), seed=9, mode="mixed"
        ),
        g0,
    )
    ids = lib.registry.ids()
    assert len(ids) == len(set(ids))
    assert lib.registry.to_json()  # round-trippable spec serialization


def test_mixed_mode_at_most_one_output_altering_defect_per_site(l0):
    log = plan_corruption(
        l0, rho=1.0, composition=composition_by_name("uniform"), seed=2024, mode="mixed"
    )
    altering = {
        CorruptionType.IMPLEMENTATION_BUG,
        CorruptionType.DOMAIN_SCOPED_BUG,
        CorruptionType.PARAM_SCHEMA_BUG,
    }
    by_site: dict[str, list[CorruptionType]] = {}
    for e in log.entries:
        if not e.is_add:
            by_site.setdefault(e.site_id, []).append(e.type)
    for types in by_site.values():
        assert sum(t in altering for t in types) <= 1


def test_generate_instance_writes_artifacts(l0, g0, battery, tmp_path):
    _lib, qa, manifest = generate_instance(
        l0,
        g0,
        rho=0.3,
        composition="uniform",
        seed=1234,
        mode="single",
        out_dir=tmp_path,
        probes=battery,
    )
    inst = tmp_path / "rho030_uniform_seed1234_single"
    for name in (
        "corruption_log.json",
        "library.json",
        "g_rho.json",
        "ideal_actions.json",
        "qa_report.json",
        "manifest.json",
    ):
        assert (inst / name).is_file()
    assert manifest.n_sites == 30
    assert qa is not None and qa.all_passed


def test_grid_config_from_yaml(tmp_path):
    p = tmp_path / "grid.yaml"
    p.write_text("rho_values: [0.2, 0.4]\ncompositions: [uniform]\nseeds: [1]\nmodes: [single]\n")
    cfg = CorruptionGridConfig.from_yaml(p)
    assert cfg.rho_values == (0.2, 0.4)
    assert cfg.seeds == (1,)


def test_plan_rejects_bad_inputs(l0):
    comp = composition_by_name("uniform")
    with pytest.raises(ValueError):
        plan_corruption(l0, rho=1.5, composition=comp, seed=1, mode="single")
    with pytest.raises(ValueError):
        plan_corruption(l0, rho=0.3, composition=comp, seed=1, mode="bogus")

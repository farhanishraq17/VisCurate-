"""``viscurate`` command-line entry point.

Intentionally small: a window into the harness for sanity checks (config, registered
skills). Heavy subcommands (build-probes, run-benchmark, curate) arrive with their phases.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from viscurate import __version__
from viscurate.config import load_config

if TYPE_CHECKING:
    from viscurate.skills.model import SkillSpec


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(cfg.model_dump_json(indent=2))
    return 0


def _cmd_skills(args: argparse.Namespace) -> int:
    # Import lazily so `viscurate --version` never pays for loading the library.
    from viscurate.skills.library import load_builtin_skills
    from viscurate.skills.registry import SkillRegistry

    registry = SkillRegistry()
    load_builtin_skills(registry)
    rows = [{"id": s.id, "name": s.name, "family": s.metadata.family} for s in registry.all()]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"{len(rows)} skills registered")
        for r in rows:
            print(f"  {r['id']:<32} [{r['family']}] {r['name']}")
    return 0


def _cmd_build_probes(args: argparse.Namespace) -> int:
    from viscurate.probes.build import ProbesConfig, build_battery

    cfg = ProbesConfig.from_yaml(args.config) if args.config else ProbesConfig()
    manifest = build_battery(cfg, args.out, args.cache, timeout=args.timeout)
    print(f"built {len(manifest)} probes -> {args.out}")
    print("domains:", manifest.domain_counts())
    print("formats:", manifest.format_counts())
    return 0


def _cmd_freeze_oracle(args: argparse.Namespace) -> int:
    from viscurate.probes.manifest import ProbeManifest
    from viscurate.probes.oracle import freeze_oracle
    from viscurate.skills.library import build_builtin_registry

    probes_dir = Path(args.probes_dir)
    manifest = ProbeManifest.model_validate_json(
        (probes_dir / "manifest.json").read_text(encoding="utf-8")
    )
    oracle = freeze_oracle(manifest, probes_dir, build_builtin_registry(), oracle_seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(oracle.model_dump_json(indent=2), encoding="utf-8")
    print(f"oracle -> {out}")
    print("status:", oracle.status_counts())
    return 0


def _cmd_build_queries(args: argparse.Namespace) -> int:
    """Phase 7 — build held-out query inputs and clean L0 reference outputs."""
    from viscurate.downstream import QueryBuildConfig, build_query_stream
    from viscurate.probes.manifest import ProbeManifest
    from viscurate.skills.library import build_builtin_registry

    cfg = QueryBuildConfig.from_yaml(args.config) if args.config else QueryBuildConfig()
    probe_manifest = None
    if not args.no_probe_check:
        manifest_path = Path(args.probes_dir) / "manifest.json"
        if manifest_path.exists():
            probe_manifest = ProbeManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        else:
            print(f"[no probe manifest at {manifest_path}] skipping probe-disjointness check")
    manifest = build_query_stream(
        cfg,
        build_builtin_registry(),
        args.out,
        probe_manifest=probe_manifest,
    )
    print(f"queries: {len(manifest)} -> {args.out}")
    print("splits:", manifest.split_counts())
    print("referenced skills:", sorted(manifest.referenced_skill_ids()))
    return 0


def _git_sha() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, timeout=5, check=False
        )
        return out.stdout.decode("ascii", "replace").strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _cmd_run_benchmark(args: argparse.Namespace) -> int:
    """Phase 4 — run the output verifier + text baselines and emit the divergence artifacts."""
    import hashlib

    from viscurate.baselines.judges import (
        EmbeddingCosineJudge,
        LlmJudge,
        NameMatchJudge,
        TextJudge,
        TfidfEmbedder,
        text_record_from_spec,
    )
    from viscurate.benchmark.ground_truth import load_ground_truth
    from viscurate.benchmark.report import write_report
    from viscurate.benchmark.runner import calibrate_from_result, run_benchmark
    from viscurate.equivalence.compare import BatteryEvaluator
    from viscurate.equivalence.param_alignment import load_param_alignment
    from viscurate.probes.build import load_probe
    from viscurate.probes.manifest import ProbeManifest
    from viscurate.skills.canonicalize import CANON_VERSION
    from viscurate.skills.library import build_builtin_registry

    cfg = load_config(args.config)
    registry = build_builtin_registry()
    skills = registry.all()
    specs = [s.to_spec() for s in skills]
    spec_by_id = {s.id: s for s in specs}

    probes_dir = Path(args.probes_dir)
    manifest_text = (probes_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = ProbeManifest.model_validate_json(manifest_text)
    probe_ids = [e.probe_id for e in manifest.entries]
    battery = [(pid, load_probe(probes_dir, pid)) for pid in probe_ids]
    provider = BatteryEvaluator(skills, battery, seed=cfg.run.seed)

    g0 = load_ground_truth(args.ground_truth, valid_ids=set(spec_by_id))
    alignment = load_param_alignment(args.param_alignment)

    embedder = TfidfEmbedder([text_record_from_spec(s).text() for s in specs])
    judges: list[TextJudge] = [NameMatchJudge(), EmbeddingCosineJudge(embedder), LlmJudge()]

    perceptual = semantic = clip = None
    if not args.no_ml:
        from viscurate.equivalence.backends import DinoBackend, LpipsBackend

        perceptual = LpipsBackend(device=args.device)
        semantic = DinoBackend(device=args.device)
        if args.clip:
            from viscurate.equivalence.backends import ClipBackend

            clip = ClipBackend(device=args.device)

    try:
        result = run_benchmark(
            specs,
            provider,
            g0,
            thresholds=cfg.thresholds,
            text_judges=judges,
            alignment=alignment,
            perceptual=perceptual,
            semantic=semantic,
            clip=clip,
            pairs=None if not args.no_ml else g0.designed_pairs(),
            screening_ids=probe_ids[: args.screening],
            candidate_k=args.k,
            compute_measurements=not args.no_measurements,
            seed=cfg.run.seed,
            meta={
                "git_sha": _git_sha(),
                "canon_version": CANON_VERSION,
                "battery_n": len(battery),
                "battery_manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
                "device": args.device,
            },
        )
    finally:
        for backend in (perceptual, semantic, clip):
            if backend is not None:
                backend.close()

    paths = write_report(result, args.out, specs=specs, manifest_extra={"config": args.config})
    print(f"benchmark: {result.meta['n_pairs']} pairs over {result.meta['n_skills']} skills")
    for name, path in paths.items():
        print(f"  {name:<16} -> {path}")
    for t in result.text_tracks:
        if not t.ran:
            print(f"  [track not run] {t.name}: {t.note}")

    if args.calibrate:
        families = (
            set(args.calib_families.split(","))
            if args.calib_families
            else _default_calibration_families(specs)
        )
        outcome = calibrate_from_result(
            result,
            spec_by_id,
            calibration_families=families,
            base=cfg.thresholds,
            date=args.date,
        )
        import yaml as _yaml

        cal_path = Path(args.out) / "calibrated_thresholds.yaml"
        cal_path.write_text(
            _yaml.safe_dump(
                {"thresholds": outcome.config.model_dump(mode="json")}, sort_keys=False
            ),
            encoding="utf-8",
        )
        print(
            f"  calibration      -> {cal_path} "
            f"(calib={outcome.n_calibration} pairs, test={outcome.n_test} pairs)"
        )
    return 0


def _cmd_corrupt(args: argparse.Namespace) -> int:
    """Phase 5 — generate the family of corrupted libraries L_ρ over the (ρ, c, seed, mode) grid."""
    from viscurate.corruption.apply import load_g0_spec
    from viscurate.corruption.grid import CorruptionGridConfig, generate_grid
    from viscurate.skills.library import build_builtin_registry
    from viscurate.skills.model import Image

    cfg = CorruptionGridConfig.from_yaml(args.config) if args.config else CorruptionGridConfig()
    l0_skills = build_builtin_registry().all()
    g0_spec = load_g0_spec(args.ground_truth)

    probes: list[tuple[str, Image]] | None = None
    if not args.no_qa:
        from viscurate.probes.build import load_probe
        from viscurate.probes.manifest import ProbeManifest

        probes_dir = Path(args.probes_dir)
        manifest_path = probes_dir / "manifest.json"
        if manifest_path.exists():
            manifest = ProbeManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            probes = [(e.probe_id, load_probe(probes_dir, e.probe_id)) for e in manifest.entries]
        else:
            print(f"[no battery at {probes_dir}] skipping QA — pass --probes-dir or --no-qa")

    manifests = generate_grid(cfg, l0_skills, g0_spec, args.out, probes=probes)
    print(f"corruption: {len(manifests)} instances -> {args.out}")
    print(f"  grid: ρ={list(cfg.rho_values)} × {list(cfg.compositions)}")
    print(f"        × {len(cfg.seeds)} seeds × modes={list(cfg.modes)}")
    if probes is not None:
        print(f"  QA over {len(probes)} probes")
    return 0


def _cmd_curate(args: argparse.Namespace) -> int:
    """Phase 6 — run the curation agent over a (clean or corrupted) library with verifier gating."""
    from viscurate.corruption.apply import apply_corruption, load_g0_spec
    from viscurate.corruption.types import CorruptionLog
    from viscurate.curation import (
        Action,
        AnthropicClient,
        CurationEnvironment,
        LlmCurationAgent,
        OllamaClient,
        ScriptedAgent,
        run_episode,
    )
    from viscurate.downstream import UsageConfig, load_query_manifest, usage_from_queries
    from viscurate.equivalence.param_alignment import load_param_alignment
    from viscurate.probes.build import load_probe
    from viscurate.probes.manifest import ProbeManifest
    from viscurate.skills.library import build_builtin_registry
    from viscurate.skills.model import Image

    cfg = load_config(args.config)

    # Library: clean L0, or an L_rho instance rebuilt from its (replayable) corruption log.
    l0_skills = build_builtin_registry().all()
    if args.instance:
        log = CorruptionLog.model_validate_json(
            (Path(args.instance) / "corruption_log.json").read_text(encoding="utf-8")
        )
        g0 = load_g0_spec(args.ground_truth)
        skills = apply_corruption(l0_skills, log, g0).registry.all()
    else:
        skills = l0_skills

    usage = None
    if args.queries_dir:
        queries = load_query_manifest(args.queries_dir)
        usage = usage_from_queries(
            queries,
            cfg=UsageConfig(
                base_count=cfg.downstream.usage_base_count,
                zipf_alpha=cfg.downstream.usage_zipf_alpha,
            ),
            registry_ids=[s.id for s in skills],
        )

    probes_dir = Path(args.probes_dir)
    manifest = ProbeManifest.model_validate_json(
        (probes_dir / "manifest.json").read_text(encoding="utf-8")
    )
    battery: list[tuple[str, Image]] = [
        (e.probe_id, load_probe(probes_dir, e.probe_id)) for e in manifest.entries
    ]

    alignment = load_param_alignment(args.param_alignment)

    # Agent selection (CLAUDE.md D7): Ollama multi-model, Claude API (optional), or a scripted
    # policy from a JSON action list; default is an end-only wiring smoke.
    if args.ollama_model:
        agent: object = LlmCurationAgent(OllamaClient(args.ollama_model, host=args.ollama_host))
    elif args.anthropic:
        agent = LlmCurationAgent(AnthropicClient(model=args.model))
    elif args.actions:
        actions = [Action.model_validate(d) for d in json.loads(Path(args.actions).read_text())]
        agent = ScriptedAgent(actions)
    else:
        agent = ScriptedAgent([])  # no policy selected → ends immediately (wiring smoke)

    perceptual = semantic = clip = None
    if not args.no_ml:
        from viscurate.equivalence.backends import DinoBackend, LpipsBackend

        perceptual = LpipsBackend(device=args.device)
        semantic = DinoBackend(device=args.device)
        if args.clip:
            from viscurate.equivalence.backends import ClipBackend

            clip = ClipBackend(device=args.device)

    try:
        env = CurationEnvironment.from_skills(
            skills,
            battery,
            thresholds=cfg.thresholds,
            seed=cfg.run.seed,
            alignment=alignment,
            perceptual=perceptual,
            semantic=semantic,
            clip=clip,
            usage=usage,
            budget=cfg.curation.budget,
            usage_fold_threshold=cfg.curation.usage_fold_threshold,
        )
        episode = run_episode(env, agent, max_steps=args.max_steps)
    finally:
        for backend in (perceptual, semantic, clip):
            if backend is not None:
                backend.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "action_log.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in episode.log], indent=2), encoding="utf-8"
    )
    (out / "episode.json").write_text(
        json.dumps(
            {
                "size_before": episode.size_before,
                "size_after": episode.size_after,
                "compression": episode.compression,
                "ended": episode.ended,
                "status_counts": episode.counts(),
                "applied_kinds": episode.applied_kinds(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"curation: {len(episode.log)} actions, {episode.size_before}->{episode.size_after} skills"
    )
    print(f"  status: {episode.counts()}")
    print(f"  applied: {episode.applied_kinds()}")
    for name, path in (("action_log", out / "action_log.json"), ("episode", out / "episode.json")):
        print(f"  {name:<12} -> {path}")
    return 0


def _cmd_run_downstream(args: argparse.Namespace) -> int:
    """Phase 7 — run a solver over the query stream and score downstream success."""
    from viscurate.corruption.apply import apply_corruption, load_g0_spec
    from viscurate.corruption.types import CorruptionLog
    from viscurate.curation import Action, CurationEnvironment, ScriptedAgent, run_episode
    from viscurate.downstream import (
        ExpectedSkillSolver,
        KeywordRetrievalSolver,
        NoOpSolver,
        SolverAgent,
        UsageConfig,
        load_query_manifest,
        run_downstream,
        usage_from_queries,
        write_downstream_report,
    )
    from viscurate.equivalence.param_alignment import load_param_alignment
    from viscurate.probes.build import load_probe
    from viscurate.probes.manifest import ProbeManifest
    from viscurate.skills.library import build_builtin_registry
    from viscurate.skills.model import Image
    from viscurate.skills.registry import SkillRegistry

    cfg = load_config(args.config)
    queries = load_query_manifest(args.queries_dir)

    l0_skills = build_builtin_registry().all()
    if args.instance:
        log = CorruptionLog.model_validate_json(
            (Path(args.instance) / "corruption_log.json").read_text(encoding="utf-8")
        )
        g0 = load_g0_spec(args.ground_truth)
        skills = apply_corruption(l0_skills, log, g0).registry.all()
        library_kind = "corrupted"
    else:
        skills = l0_skills
        library_kind = "clean"

    usage = usage_from_queries(
        queries,
        cfg=UsageConfig(
            base_count=cfg.downstream.usage_base_count,
            zipf_alpha=cfg.downstream.usage_zipf_alpha,
        ),
        registry_ids=[s.id for s in skills],
    )

    perceptual = semantic = clip = None
    if not args.no_ml:
        from viscurate.equivalence.backends import DinoBackend, LpipsBackend

        perceptual = LpipsBackend(device=args.device)
        semantic = DinoBackend(device=args.device)
        if args.clip:
            from viscurate.equivalence.backends import ClipBackend

            clip = ClipBackend(device=args.device)

    try:
        registry = SkillRegistry()
        for skill in skills:
            registry.register(skill)
        curation_episode = None
        if args.actions:
            probes_dir = Path(args.probes_dir)
            manifest = ProbeManifest.model_validate_json(
                (probes_dir / "manifest.json").read_text(encoding="utf-8")
            )
            battery: list[tuple[str, Image]] = [
                (e.probe_id, load_probe(probes_dir, e.probe_id)) for e in manifest.entries
            ]
            actions = [Action.model_validate(d) for d in json.loads(Path(args.actions).read_text())]
            env = CurationEnvironment.from_skills(
                skills,
                battery,
                thresholds=cfg.thresholds,
                seed=cfg.run.seed,
                alignment=load_param_alignment(args.param_alignment),
                perceptual=perceptual,
                semantic=semantic,
                clip=clip,
                usage=usage,
                budget=cfg.curation.budget,
                usage_fold_threshold=cfg.curation.usage_fold_threshold,
            )
            curation_episode = run_episode(env, ScriptedAgent(actions), max_steps=args.max_steps)
            registry = env.registry
            library_kind = f"{library_kind}+scripted-curation"

        solver: SolverAgent
        if args.solver == "expected":
            solver = ExpectedSkillSolver()
        elif args.solver == "keyword":
            solver = KeywordRetrievalSolver()
        elif args.solver == "noop":
            solver = NoOpSolver()
        else:  # pragma: no cover - argparse choices guard this
            raise ValueError(args.solver)

        splits = tuple(s.strip() for s in args.splits.split(",") if s.strip()) or None
        result = run_downstream(
            queries,
            args.queries_dir,
            registry,
            solver,
            thresholds=cfg.thresholds,
            perceptual=perceptual,
            seed=cfg.run.seed,
            splits=splits,
            meta={
                "library_kind": library_kind,
                "instance": args.instance,
                "actions": args.actions,
                "curation_compression": None
                if curation_episode is None
                else curation_episode.compression,
            },
        )
    finally:
        for backend in (perceptual, semantic, clip):
            if backend is not None:
                backend.close()

    paths = write_downstream_report(
        result,
        args.out,
        manifest_extra={"config": args.config, "queries_dir": args.queries_dir},
    )
    print(
        f"downstream: {result.n} queries, success={result.success_rate():.3f}, "
        f"solver={result.meta['solver']}"
    )
    for split in sorted({s.split for s in result.scores}):
        n_split = sum(1 for s in result.scores if s.split == split)
        print(f"  {split}: {result.success_rate(split):.3f} over {n_split} queries")
    for name, path in paths.items():
        print(f"  {name:<12} -> {path}")
    return 0


def _cmd_phase8(args: argparse.Namespace) -> int:
    """Phase 8 — aggregate study rows into Pareto/correlation/ablation artifacts."""
    from viscurate.studies import load_study_points, write_study_report

    points = load_study_points(args.points)
    if not points:
        raise ValueError("no study points supplied; Phase-8 reports only aggregate real rows")
    paths = write_study_report(
        points,
        args.out,
        title=args.title,
        output_gate=args.output_gate,
        text_gate=args.text_gate,
        manifest_extra={"points": args.points, "git_sha": _git_sha()},
    )
    print(f"phase8: {len(points)} seed-level points -> {args.out}")
    print("  methods:", ", ".join(sorted({p.method for p in points})))
    for name, path in paths.items():
        print(f"  {name:<18} -> {path}")
    return 0


def _cmd_phase9(args: argparse.Namespace) -> int:
    """Phase 9 — write the manifest-backed reproducibility + paper-artifact bundle."""
    from viscurate.experiments import Phase9Config, run_phase9

    cfg = Phase9Config.from_yaml(args.config)
    result = run_phase9(cfg, args.out, config_path=args.config)
    print(f"phase9: reproducibility bundle -> {result.out_dir}")
    for name, path in (
        ("run_manifest", result.manifest),
        ("realism_audit", result.audit_markdown),
        ("audit_json", result.audit_json),
        ("repro_script", result.repro_script),
        ("config_snapshot", result.config_snapshot),
    ):
        print(f"  {name:<16} -> {path}")
    if result.paper_artifacts_dir is not None:
        print(f"  paper_artifacts  -> {result.paper_artifacts_dir}")
    else:
        print("  paper_artifacts  -> pending (no real StudyPoint file configured)")
    return 0


def _default_calibration_families(specs: Sequence[SkillSpec]) -> set[str]:
    """A deterministic, documented cluster-disjoint split: the alphabetically-first half."""
    families = sorted({s.metadata.family for s in specs})
    return set(families[: len(families) // 2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="viscurate", description="VisCurate harness CLI")
    parser.add_argument("--version", action="version", version=f"viscurate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_config = sub.add_parser("config", help="load + validate a YAML config and print it")
    p_config.add_argument("-c", "--config", default=None, help="path to YAML config")
    p_config.set_defaults(func=_cmd_config)

    p_skills = sub.add_parser("skills", help="list registered built-in skills")
    p_skills.add_argument("--json", action="store_true", help="emit JSON")
    p_skills.set_defaults(func=_cmd_skills)

    p_probes = sub.add_parser("build-probes", help="build the probe battery + manifest")
    p_probes.add_argument("-c", "--config", default="configs/probes.yaml", help="probes YAML")
    p_probes.add_argument("-o", "--out", default="data/probe_images", help="output dir")
    p_probes.add_argument("--cache", default="data/cache", help="metadata cache dir")
    p_probes.add_argument("--timeout", type=float, default=30.0, help="download timeout (s)")
    p_probes.set_defaults(func=_cmd_build_probes)

    p_oracle = sub.add_parser("freeze-oracle", help="freeze the reference oracle over the battery")
    p_oracle.add_argument("--probes-dir", default="data/probe_images", help="battery dir")
    p_oracle.add_argument("-o", "--out", default="data/oracle/oracle.json", help="oracle manifest")
    p_oracle.add_argument("--seed", type=int, default=0, help="oracle execution seed")
    p_oracle.set_defaults(func=_cmd_freeze_oracle)

    p_queries = sub.add_parser(
        "build-queries", help="Phase 7: build downstream query inputs + references"
    )
    p_queries.add_argument("-c", "--config", default="configs/queries.yaml", help="queries YAML")
    p_queries.add_argument("-o", "--out", default="data/queries", help="output dir")
    p_queries.add_argument("--probes-dir", default="data/probe_images", help="probe dir to avoid")
    p_queries.add_argument(
        "--no-probe-check", action="store_true", help="skip query/probe hash disjointness check"
    )
    p_queries.set_defaults(func=_cmd_build_queries)

    p_bench = sub.add_parser(
        "run-benchmark", help="Phase 4: run the equivalence benchmark + emit the divergence report"
    )
    p_bench.add_argument("-c", "--config", default=None, help="path to YAML config (thresholds)")
    p_bench.add_argument("--probes-dir", default="data/probe_images", help="battery dir")
    p_bench.add_argument(
        "--ground-truth", default="configs/ground_truth_g0.yaml", help="designed relation graph G0"
    )
    p_bench.add_argument(
        "--param-alignment", default="configs/param_alignment.yaml", help="matched-sweep axes"
    )
    p_bench.add_argument("-o", "--out", default="results/phase4_benchmark", help="output dir")
    p_bench.add_argument(
        "--device", default="cpu", help="torch device for ML backends (e.g. cuda for the H200)"
    )
    p_bench.add_argument(
        "--no-ml", action="store_true", help="skip ML backends (EXACT/SUBSUMPTION + text only)"
    )
    p_bench.add_argument("--clip", action="store_true", help="add CLIP as a 2nd semantic view")
    p_bench.add_argument("-k", type=int, default=5, dest="k", help="candidate NN fan-out")
    p_bench.add_argument("--screening", type=int, default=12, help="screening sub-battery size")
    p_bench.add_argument(
        "--no-measurements", action="store_true", help="skip full distance vectors (faster)"
    )
    p_bench.add_argument("--calibrate", action="store_true", help="calibrate thresholds on the run")
    p_bench.add_argument(
        "--calib-families", default="", help="comma-separated families for the calibration cluster"
    )
    p_bench.add_argument(
        "--date", default="", help="calibration date stamp (YYYY-MM-DD) when --calibrate"
    )
    p_bench.set_defaults(func=_cmd_run_benchmark)

    p_corrupt = sub.add_parser(
        "corrupt", help="Phase 5: generate the family of corrupted libraries L_ρ + ground-truth"
    )
    p_corrupt.add_argument("-c", "--config", default="configs/corruption.yaml", help="grid YAML")
    p_corrupt.add_argument(
        "--ground-truth", default="configs/ground_truth_g0.yaml", help="designed relation graph G0"
    )
    p_corrupt.add_argument("-o", "--out", default="data/corruption", help="output dir")
    p_corrupt.add_argument("--probes-dir", default="data/probe_images", help="battery dir (for QA)")
    p_corrupt.add_argument(
        "--no-qa", action="store_true", help="skip the per-defect QA assertions (no battery needed)"
    )
    p_corrupt.set_defaults(func=_cmd_corrupt)

    p_curate = sub.add_parser(
        "curate", help="Phase 6: run the curation agent over a library with verifier gating"
    )
    p_curate.add_argument("-c", "--config", default=None, help="path to YAML config")
    p_curate.add_argument("--probes-dir", default="data/probe_images", help="battery dir")
    p_curate.add_argument(
        "--instance", default="", help="corruption instance dir to curate (default: clean L0)"
    )
    p_curate.add_argument(
        "--ground-truth", default="configs/ground_truth_g0.yaml", help="G0 (for --instance replay)"
    )
    p_curate.add_argument(
        "--param-alignment", default="configs/param_alignment.yaml", help="matched-sweep axes"
    )
    p_curate.add_argument("-o", "--out", default="results/phase6_curation", help="output dir")
    p_curate.add_argument("--device", default="cpu", help="torch device for ML backends")
    p_curate.add_argument("--no-ml", action="store_true", help="skip ML backends (EXACT/SUB only)")
    p_curate.add_argument("--clip", action="store_true", help="add CLIP as a 2nd semantic view")
    p_curate.add_argument("--max-steps", type=int, default=None, help="cap on agent steps")
    p_curate.add_argument("--actions", default="", help="JSON file of scripted actions to replay")
    p_curate.add_argument(
        "--queries-dir", default="", help="Phase-7 query dir for query-derived UsageStats"
    )
    p_curate.add_argument("--ollama-model", default="", help="drive the agent with an Ollama model")
    p_curate.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL")
    p_curate.add_argument("--anthropic", action="store_true", help="drive the agent via Claude API")
    p_curate.add_argument(
        "--model", default="claude-opus-4-8", help="Claude model id (with --anthropic)"
    )
    p_curate.set_defaults(func=_cmd_curate)

    p_down = sub.add_parser(
        "run-downstream", help="Phase 7: score a solver over the downstream query stream"
    )
    p_down.add_argument("-c", "--config", default=None, help="path to YAML config")
    p_down.add_argument("--queries-dir", default="data/queries", help="query stream dir")
    p_down.add_argument(
        "--instance", default="", help="corruption instance dir to evaluate (default: clean L0)"
    )
    p_down.add_argument(
        "--ground-truth", default="configs/ground_truth_g0.yaml", help="G0 (for --instance replay)"
    )
    p_down.add_argument(
        "--actions", default="", help="optional scripted curation actions to replay before scoring"
    )
    p_down.add_argument(
        "--probes-dir", default="data/probe_images", help="battery dir for curation"
    )
    p_down.add_argument(
        "--param-alignment", default="configs/param_alignment.yaml", help="matched-sweep axes"
    )
    p_down.add_argument(
        "--solver",
        choices=("expected", "keyword", "noop"),
        default="expected",
        help="solver policy to evaluate",
    )
    p_down.add_argument("--splits", default="", help="comma-separated splits to score")
    p_down.add_argument("-o", "--out", default="results/phase7_downstream", help="output dir")
    p_down.add_argument(
        "--device", default="cpu", help="torch device for optional ML scoring/gates"
    )
    p_down.add_argument("--no-ml", action="store_true", help="skip LPIPS/DINO backends")
    p_down.add_argument("--clip", action="store_true", help="add CLIP for curation verifier")
    p_down.add_argument("--max-steps", type=int, default=None, help="cap scripted curation steps")
    p_down.set_defaults(func=_cmd_run_downstream)

    p_phase8 = sub.add_parser(
        "phase8", help="Phase 8: aggregate study rows into tables, Pareto front, and ablations"
    )
    p_phase8.add_argument(
        "--points",
        required=True,
        help="JSON or CSV of seed-level StudyPoint rows (real run artifacts summarized)",
    )
    p_phase8.add_argument("-o", "--out", default="results/phase8_studies", help="output dir")
    p_phase8.add_argument(
        "--title", default="VisCurate — Phase 8 Studies", help="markdown report title"
    )
    p_phase8.add_argument(
        "--output-gate", default="output", help="gate label for output-gated ablation rows"
    )
    p_phase8.add_argument(
        "--text-gate", default="text", help="gate label for text-gated ablation rows"
    )
    p_phase8.set_defaults(func=_cmd_phase8)

    p_phase9 = sub.add_parser(
        "phase9", help="Phase 9: write run manifests, realism audit, and paper artifacts"
    )
    p_phase9.add_argument(
        "-c", "--config", default="configs/phase9.yaml", help="Phase-9 experiment YAML"
    )
    p_phase9.add_argument("-o", "--out", default="results/phase9", help="output dir")
    p_phase9.set_defaults(func=_cmd_phase9)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

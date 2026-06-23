"""``viscurate`` command-line entry point.

Intentionally small: a window into the harness for sanity checks (config, registered
skills). Heavy subcommands (build-probes, run-benchmark, curate) arrive with their phases.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from viscurate import __version__
from viscurate.config import load_config


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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

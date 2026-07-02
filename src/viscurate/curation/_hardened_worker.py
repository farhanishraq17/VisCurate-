"""Child process entry point for hardened agent-authored skill execution."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 5:
        print("usage: _hardened_worker SOURCE INPUT PARAMS SEED OUTPUT", file=sys.stderr)
        return 2
    source_path, input_path, params_path, seed_text, output_path = args
    spec = importlib.util.spec_from_file_location("viscurate_agent_skill", source_path)
    if spec is None or spec.loader is None:
        print("cannot load source", file=sys.stderr)
        return 2
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        print("source must define run(image, params, seed)", file=sys.stderr)
        return 2
    image = np.load(input_path, allow_pickle=False)
    params: dict[str, Any] = json.loads(Path(params_path).read_text(encoding="utf-8"))
    out = run(image, params, int(seed_text))
    if not isinstance(out, np.ndarray):
        print(f"run returned {type(out).__name__}, expected ndarray", file=sys.stderr)
        return 2
    np.save(output_path, out, allow_pickle=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Subprocess worker for :class:`viscurate.skills.executor.SandboxedExecutor`.

Invoked as ``python -m viscurate.skills._worker <in.pkl> <out.pkl>``. Reads a pickled
request ``{skill, image, params, seed}``, runs the skill in-process *in this isolated
child*, and writes back ``{ok, output|error, traceback}``.

Only **trusted** skills ever reach this worker — the executor blocks ``trusted=False``
skills before spawning a child (CLAUDE.md §5), so unpickling our own callables here is
safe. Agent-generated code waits for the Phase-6 hardened sandbox.
"""

from __future__ import annotations

import pickle
import sys
import traceback
from typing import Any


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write("usage: _worker <input.pkl> <output.pkl>\n")
        return 2
    in_path, out_path = argv[1], argv[2]

    with open(in_path, "rb") as fh:
        req: dict[str, Any] = pickle.load(fh)

    result: dict[str, Any]
    try:
        skill = req["skill"]
        out = skill.run(req["image"], req.get("params"), int(req.get("seed", 0)))
        result = {"ok": True, "output": out}
    except BaseException as exc:
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    with open(out_path, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

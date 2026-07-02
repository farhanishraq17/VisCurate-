"""Hardened subprocess runner for agent-authored skill functions.

This is intentionally separate from the lightweight trusted-skill executor. Agent source is
written to a temporary file and imported only by a child process. When `bwrap` is available the
child runs without network access and with only the Python environment, repository source, and
per-run work directory mounted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from viscurate.config import ExecutorConfig
from viscurate.skills.model import Image, Params, SkillFn

__all__ = ["HardenedExecutor", "HardenedRunResult", "make_hardened_fn"]


@dataclass(frozen=True)
class HardenedRunResult:
    ok: bool
    output: Image | None
    error: str
    duration_s: float
    timed_out: bool = False


class HardenedExecutor:
    """Execute agent-authored Python `run(image, params, seed)` functions in a child sandbox."""

    def __init__(
        self,
        config: ExecutorConfig | None = None,
        *,
        python: str | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config = config or ExecutorConfig()
        self.python = python or sys.executable
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[3])
        self.bwrap = shutil.which("bwrap")

    @property
    def available(self) -> bool:
        return self.bwrap is not None or sys.platform == "win32"

    def run_source(
        self, source: str, image: Image, params: Params | None = None, seed: int = 0
    ) -> HardenedRunResult:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="viscurate_hardened_") as tmp:
            work = Path(tmp)
            src = work / "skill_source.py"
            inp = work / "input.npy"
            params_path = work / "params.json"
            out = work / "output.npy"
            src.write_text(source, encoding="utf-8")
            np.save(inp, image, allow_pickle=False)
            params_path.write_text(json.dumps(params or {}, sort_keys=True), encoding="utf-8")
            cmd = self._command(work, src, inp, params_path, out, seed)
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONNOUSERSITE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            try:
                proc = subprocess.run(
                    cmd,
                    timeout=self.config.timeout_s,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return HardenedRunResult(
                    ok=False,
                    output=None,
                    error=f"TIMEOUT after {self.config.timeout_s}s",
                    duration_s=time.perf_counter() - start,
                    timed_out=True,
                )
            duration = time.perf_counter() - start
            if proc.returncode != 0 or not out.exists():
                err = (proc.stderr or proc.stdout or "").strip()
                return HardenedRunResult(
                    ok=False,
                    output=None,
                    error=f"worker exited {proc.returncode}: {err[-500:]}",
                    duration_s=duration,
                )
            try:
                arr = np.load(out, allow_pickle=False)
            except Exception as exc:
                return HardenedRunResult(
                    ok=False,
                    output=None,
                    error=f"invalid output: {type(exc).__name__}: {exc}",
                    duration_s=duration,
                )
            return HardenedRunResult(ok=True, output=arr, error="", duration_s=duration)

    def _command(
        self,
        work: Path,
        source: Path,
        input_path: Path,
        params_path: Path,
        output_path: Path,
        seed: int,
    ) -> list[str]:
        worker_args = [
            self.python,
            "-m",
            "viscurate.curation._hardened_worker",
            str(source),
            str(input_path),
            str(params_path),
            str(seed),
            str(output_path),
        ]
        if self.bwrap is None:
            return worker_args
        env_root = Path(self.python).resolve().parents[1]
        return [
            self.bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--ro-bind",
            str(env_root),
            str(env_root),
            "--ro-bind",
            str(self.repo_root / "src"),
            str(self.repo_root / "src"),
            "--bind",
            str(work),
            str(work),
            "--setenv",
            "PYTHONPATH",
            str(self.repo_root / "src"),
            "--",
            *worker_args,
        ]


def make_hardened_fn(source: str, executor: HardenedExecutor) -> SkillFn:
    def _fn(image: Image, params: Params, seed: int) -> Image:
        result = executor.run_source(source, image, params, seed)
        if not result.ok or result.output is None:
            raise RuntimeError(result.error)
        return result.output

    return _fn

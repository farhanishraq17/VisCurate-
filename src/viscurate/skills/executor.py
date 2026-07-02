"""Lightweight sandboxed executor (CLAUDE.md D6, Phase 1).

Each skill runs in a **fresh subprocess** with a wall-clock timeout, so a hanging or
crashing skill returns a structured error instead of taking down the harness. On POSIX
(the WSL2 dev target) we additionally apply ``RLIMIT_AS``/``RLIMIT_CPU`` via ``preexec_fn``;
on Windows ``resource`` is unavailable, so we degrade to timeout-only and log it once.

This is *lightweight* isolation appropriate for the trusted 100-skill starter set. The
hardened sandbox (network namespace, restricted FS) for agent-generated code is deferred
to Phase 6; ``trusted=False`` skills are blocked here and never spawn a child.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from viscurate.config import ExecutorConfig
from viscurate.logging import get_logger
from viscurate.skills.model import Image, Params, Skill

__all__ = ["ExecutionResult", "SandboxedExecutor"]

_IS_POSIX = os.name == "posix"
_WORKER_ENV_LIMITS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one sandboxed execution."""

    ok: bool
    output: Image | None
    error: str | None
    duration_s: float
    timed_out: bool = False
    blocked: bool = False
    returncode: int | None = None


def _posix_preexec(max_memory_mb: int | None, cpu_s: int) -> Callable[[], None] | None:
    """Return a ``preexec_fn`` that sets address-space and CPU rlimits (POSIX only)."""
    # `sys.platform` (not os.name) so a type checker on Windows treats the body below as
    # unreachable — the `resource` module is POSIX-only — while it is still checked on the
    # WSL2 dev target.
    if sys.platform == "win32":
        return None
    import resource

    # Darwin accepts the `resource` module but can reject RLIMIT_AS; keep the executor usable
    # there with CPU+wall-time limits rather than failing before the worker starts.
    set_memory = max_memory_mb is not None and sys.platform != "darwin"

    def _apply() -> None:  # pragma: no cover - runs only in POSIX child
        if set_memory:
            assert max_memory_mb is not None
            nbytes = max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))

    return _apply


class SandboxedExecutor:
    """Runs trusted skills in isolated subprocesses; logs every execution."""

    def __init__(self, config: ExecutorConfig | None = None, *, logger: Any | None = None) -> None:
        self._config = config or ExecutorConfig()
        self._log = logger or get_logger("executor")
        if not _IS_POSIX:
            self._log.warning(
                "executor_rlimits_unavailable",
                platform=sys.platform,
                detail="resource module is POSIX-only; falling back to wall-clock timeout only",
            )
        elif sys.platform == "darwin" and self._config.max_memory_mb is not None:
            self._log.warning(
                "executor_memory_rlimit_unavailable",
                platform=sys.platform,
                detail="RLIMIT_AS is unreliable on Darwin; using CPU rlimit + wall-clock timeout",
            )

    @property
    def config(self) -> ExecutorConfig:
        return self._config

    def run(
        self, skill: Skill, image: Image, params: Params | None = None, seed: int = 0
    ) -> ExecutionResult:
        """Execute ``skill`` on ``image`` in a sandboxed child process."""
        # Hard trusted gate — never spawn a child for untrusted code (CLAUDE.md §5).
        if not skill.metadata.trusted and not self._config.allow_untrusted:
            self._log.warning("execution_blocked", skill_id=skill.id, reason="untrusted")
            return ExecutionResult(
                ok=False,
                output=None,
                error="BLOCKED: skill is not trusted and allow_untrusted is False",
                duration_s=0.0,
                blocked=True,
            )

        with tempfile.TemporaryDirectory(prefix="viscurate_exec_") as tmp:
            in_path = Path(tmp) / "in.pkl"
            out_path = Path(tmp) / "out.pkl"
            with in_path.open("wb") as fh:
                pickle.dump(
                    {"skill": skill, "image": image, "params": params, "seed": int(seed)},
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            cmd = [sys.executable, "-m", "viscurate.skills._worker", str(in_path), str(out_path)]
            cpu_budget = max(1, int(self._config.timeout_s) + 1)
            preexec = _posix_preexec(self._config.max_memory_mb, cpu_budget)
            env = {**os.environ, **_WORKER_ENV_LIMITS}
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    timeout=self._config.timeout_s,
                    capture_output=True,
                    preexec_fn=preexec,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                duration = time.perf_counter() - start
                self._log.warning(
                    "execution_timeout", skill_id=skill.id, timeout_s=self._config.timeout_s
                )
                return ExecutionResult(
                    ok=False,
                    output=None,
                    error=f"TIMEOUT after {self._config.timeout_s}s",
                    duration_s=duration,
                    timed_out=True,
                )
            except subprocess.SubprocessError as exc:
                if preexec is None or "preexec_fn" not in str(exc):
                    duration = time.perf_counter() - start
                    self._log.warning("execution_failed", skill_id=skill.id, error=str(exc))
                    return ExecutionResult(
                        ok=False,
                        output=None,
                        error=f"worker launch failed: {exc}",
                        duration_s=duration,
                    )
                self._log.warning(
                    "execution_rlimit_degraded",
                    skill_id=skill.id,
                    detail="preexec_fn failed; retrying once without POSIX rlimits",
                    error=str(exc),
                )
                proc = subprocess.run(
                    cmd,
                    timeout=self._config.timeout_s,
                    capture_output=True,
                    preexec_fn=None,
                    env=env,
                )
            duration = time.perf_counter() - start

            if proc.returncode != 0 or not out_path.exists():
                stderr = proc.stderr.decode("utf-8", "replace").strip()
                self._log.warning("execution_failed", skill_id=skill.id, returncode=proc.returncode)
                return ExecutionResult(
                    ok=False,
                    output=None,
                    error=f"worker exited {proc.returncode}: {stderr[-500:]}",
                    duration_s=duration,
                    returncode=proc.returncode,
                )

            with out_path.open("rb") as fh:
                result: dict[str, Any] = pickle.load(fh)

        if result.get("ok"):
            self._log.info("execution_ok", skill_id=skill.id, duration_s=round(duration, 4))
            return ExecutionResult(
                ok=True, output=result["output"], error=None, duration_s=duration, returncode=0
            )

        self._log.warning("execution_error", skill_id=skill.id, error=result.get("error"))
        return ExecutionResult(
            ok=False,
            output=None,
            error=str(result.get("error")),
            duration_s=duration,
            returncode=0,
        )

"""A4 — the telemetry layer. BUILD THIS FIRST; every other experiment depends on it.

DESTINATION: ``src/viscurate/instrument/telemetry.py`` in the VisCurate repo
             (plus ``src/viscurate/instrument/__init__.py`` re-exporting ``recorder``).

WHY FIRST
---------
A4 is fourth by *value* but first by *dependency*. Timing, token, cache, and memory data cannot
be reconstructed after a run — if the Phase-4 GPU run happens without these hooks, it produces
no cost data and has to be repeated. Wire this in before any GPU time is spent.

WHAT IT MEASURES, AND THE TWO FIELDS THAT CARRY THE WHOLE ARGUMENT
------------------------------------------------------------------
``cache_hit`` on ``signature_compute``
    Proves signature computation is O(n) rather than O(n²) *in this workload*. Note this is
    evidence about constants, NOT a proof of the asymptotic bound — prove the bound from the
    implementation and use telemetry to confirm no recomputation occurs.

``deciding_stage`` on ``pair_verify``
    ``taxonomy.classify`` is hierarchical and stop-at-first. Recording which stage decided each
    pair shows how often the expensive learned backends are reached at all.

    ⚠ CAVEAT confirmed by reading ``equivalence/backends.py`` + ``compare.py``: the design is
    **eager-per-stage with output caching** — ``BatteryEvaluator`` caches outputs per
    ``(skill, params, seed)`` and backends batch-extract features per stage before ``close()``.
    So short-circuiting saves *distance computation*, not the backbone forward passes, which are
    already paid. Do NOT claim both "features are cached" and "short-circuiting avoids forward
    passes" — they are in tension. Report the deciding-stage distribution as an interesting
    property; do not convert it into a forward-pass saving.

GPU TIMING IS NOT ``time.time()``
---------------------------------
CUDA is asynchronous: naive wall-clock around a kernel launch measures the *launch*, not the
work. ``gpu_timer`` below uses CUDA events with an explicit synchronize. Always discard warm-up
iterations and report median AND p95 — tail latency is what a practitioner feels, especially on
real corpora with heavy-tailed runtimes.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "NullRecorder",
    "Recorder",
    "RunManifest",
    "active_recorder",
    "gpu_timer",
    "peak_gpu_memory_bytes",
    "recorder",
    "timer",
]


# ==============================================================================================
# Records
# ==============================================================================================


@dataclass(frozen=True)
class Event:
    """One telemetry record. ``kind`` selects the schema of ``fields``."""

    kind: str
    fields: dict[str, Any]
    t_wall: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RunManifest:
    """Environment metadata recorded once per run. Timing numbers are meaningless without it."""

    run_id: str
    git_sha: str
    python: str
    platform: str
    hostname: str
    gpu_name: str
    gpu_total_mem_bytes: int
    torch_version: str
    cuda_version: str
    started_at: float

    @staticmethod
    def capture(run_id: str | None = None) -> RunManifest:
        gpu_name, gpu_mem, torch_v, cuda_v = _torch_env()
        return RunManifest(
            run_id=run_id or uuid.uuid4().hex[:12],
            git_sha=_git_sha(),
            python=platform.python_version(),
            platform=platform.platform(),
            hostname=platform.node(),
            gpu_name=gpu_name,
            gpu_total_mem_bytes=gpu_mem,
            torch_version=torch_v,
            cuda_version=cuda_v,
            started_at=time.time(),
        )


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - environment guard
        return "unknown"


def _torch_env() -> tuple[str, int, str, str]:
    try:
        import torch
    except ImportError:
        return ("none", 0, "not-installed", "none")
    if not torch.cuda.is_available():
        return ("cpu", 0, torch.__version__, "none")
    props = torch.cuda.get_device_properties(0)
    return (props.name, int(props.total_memory), torch.__version__, torch.version.cuda or "none")


# ==============================================================================================
# The recorder
# ==============================================================================================


class Recorder:
    """Append-only JSONL telemetry sink.

    Deliberately *not* gated behind a verbosity flag — a flag someone forgets to set is how a
    12-hour GPU run comes back with no cost data. Emit always; the file is tiny next to features.
    """

    def __init__(self, out_path: str | Path, manifest: RunManifest | None = None) -> None:
        self.path = Path(out_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest or RunManifest.capture()
        self._fh = self.path.open("a", encoding="utf-8")
        self._write(
            {"kind": "run_manifest", "fields": asdict(self.manifest), "t_wall": time.time()}
        )

    def _write(self, obj: dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj, default=str) + "\n")
        self._fh.flush()  # a crashed run must not lose its telemetry

    def emit(self, kind: str, **fields: Any) -> None:
        self._write({"kind": kind, "fields": fields, "t_wall": time.time()})

    # -- typed helpers: one per event class in A4/01_measurement_protocol.md §1.1 ---------------

    def signature_compute(
        self,
        *,
        skill_id: str,
        n_probes: int,
        n_grid: int,
        wall_ms: float,
        gpu_ms: float,
        peak_mem: int,
        backend: str,
        cache_hit: bool,
    ) -> None:
        self.emit(
            "signature_compute",
            skill_id=skill_id,
            n_probes=n_probes,
            n_grid=n_grid,
            wall_ms=wall_ms,
            gpu_ms=gpu_ms,
            peak_mem=peak_mem,
            backend=backend,
            cache_hit=cache_hit,
        )

    def pair_verify(
        self,
        *,
        pair_id: str,
        deciding_stage: str,
        wall_ms: float,
        gpu_ms: float,
        verdict: str,
        short_circuited: bool,
    ) -> None:
        self.emit(
            "pair_verify",
            pair_id=pair_id,
            deciding_stage=deciding_stage,
            wall_ms=wall_ms,
            gpu_ms=gpu_ms,
            verdict=verdict,
            short_circuited=short_circuited,
        )

    def candidate_gen(
        self, *, n_skills: int, radius: float, n_candidates: int, n_all_pairs: int, wall_ms: float
    ) -> None:
        self.emit(
            "candidate_gen",
            n_skills=n_skills,
            radius=radius,
            n_candidates=n_candidates,
            n_all_pairs=n_all_pairs,
            reduction=1.0 - (n_candidates / max(n_all_pairs, 1)),
            wall_ms=wall_ms,
        )

    def llm_call(
        self,
        *,
        model: str,
        tokens_in: int,
        tokens_out: int,
        effort: str,
        wall_ms: float,
        retry_count: int = 0,
    ) -> None:
        """⚠ Record tokens and latency. Do NOT record dollars here — prices change and a run
        must stay reproducible. Convert to USD at analysis time with a timestamped price table."""
        self.emit(
            "llm_call",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            effort=effort,
            wall_ms=wall_ms,
            retry_count=retry_count,
        )

    def agent_episode(
        self,
        *,
        model: str,
        instance_id: str,
        n_actions: int,
        n_applied: int,
        n_rejected: int,
        n_blocked: int,
        n_invalid: int,
        tokens_in: int,
        tokens_out: int,
        wall_ms: float,
    ) -> None:
        self.emit(
            "agent_episode",
            model=model,
            instance_id=instance_id,
            n_actions=n_actions,
            n_applied=n_applied,
            n_rejected=n_rejected,
            n_blocked=n_blocked,
            n_invalid=n_invalid,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            wall_ms=wall_ms,
        )

    def feature_cache(self, *, n_entries: int, bytes_resident: int, dim: int, dtype: str) -> None:
        """Memory may be the binding constraint at scale, not compute: at n=1e4 with 177 probes
        x 5 grid points x 768-dim float32, the DINO cache alone is ~27 GB (~80 GB with LPIPS and
        CLIP; ~800 GB at n=1e5). Record it so the extrapolation is honest."""
        self.emit(
            "feature_cache",
            n_entries=n_entries,
            bytes_resident=bytes_resident,
            dim=dim,
            dtype=dtype,
        )

    def close(self) -> None:
        self.emit("run_end", duration_s=time.time() - self.manifest.started_at)
        self._fh.close()


class NullRecorder:
    """No-op sink used when ``VISCURATE_TELEMETRY`` is unset.

    Instrumentation is wired unconditionally into the pipeline (a flag someone forgets to set is
    how a 12-hour GPU run comes back with no cost data), so the *call sites* must be free of
    conditionals. The opt-out lives here instead: without the env var the pipeline behaves
    exactly as before and writes no file, which keeps the test suite from littering artifacts.
    """

    path = None
    manifest = None

    def emit(self, kind: str, **fields: Any) -> None:
        return None

    def signature_compute(self, **fields: Any) -> None:
        return None

    def pair_verify(self, **fields: Any) -> None:
        return None

    def candidate_gen(self, **fields: Any) -> None:
        return None

    def llm_call(self, **fields: Any) -> None:
        return None

    def agent_episode(self, **fields: Any) -> None:
        return None

    def feature_cache(self, **fields: Any) -> None:
        return None

    def close(self) -> None:
        return None


_ACTIVE: Recorder | None = None
_NULL = NullRecorder()


def recorder(out_path: str | Path | None = None) -> Recorder:
    """Process-global recorder. Call once at run start; retrieve anywhere afterwards."""
    global _ACTIVE
    if out_path is not None or _ACTIVE is None:
        path = out_path or os.environ.get("VISCURATE_TELEMETRY", "results/telemetry.jsonl")
        _ACTIVE = Recorder(path)
    return _ACTIVE


def active_recorder() -> Recorder | NullRecorder:
    """The recorder to emit into from library code.

    Returns the process-global :class:`Recorder` once one exists, opens one on first use when
    ``VISCURATE_TELEMETRY`` names a path, and otherwise returns a :class:`NullRecorder`. Library
    call sites use this rather than :func:`recorder` so instrumentation is always wired but never
    forces an artifact on a caller that did not ask for telemetry.
    """
    if _ACTIVE is not None:
        return _ACTIVE
    if os.environ.get("VISCURATE_TELEMETRY"):
        return recorder()
    return _NULL


# ==============================================================================================
# Timers
# ==============================================================================================


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Wall-clock timer. ``out["ms"]`` is populated on exit."""
    out: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["ms"] = (time.perf_counter() - t0) * 1000.0


@contextmanager
def gpu_timer(device: str = "cuda") -> Iterator[dict[str, float]]:
    """CUDA-event timer with explicit synchronize.

    Falls back to wall-clock on CPU. Populates ``out["gpu_ms"]`` and ``out["wall_ms"]``.

    Usage — ALWAYS discard warm-up iterations before recording::

        for i in range(n_warmup + n_trials):
            with gpu_timer() as t:
                run_stage()
            if i >= n_warmup:
                samples.append(t["gpu_ms"])
        # report median AND p95, with an interval — not a single mean
    """
    out: dict[str, float] = {}
    try:
        import torch

        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    except ImportError:
        use_cuda = False

    if not use_cuda:
        t0 = time.perf_counter()
        try:
            yield out
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            out["wall_ms"] = ms
            out["gpu_ms"] = 0.0
        return

    import torch

    start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    try:
        yield out
    finally:
        end.record()
        torch.cuda.synchronize()  # REQUIRED — without this you time the launch, not the work
        out["gpu_ms"] = float(start.elapsed_time(end))
        out["wall_ms"] = (time.perf_counter() - t0) * 1000.0


def peak_gpu_memory_bytes(reset: bool = False) -> int:
    """Peak allocated GPU memory. Measure it; never infer it from model size."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        peak = int(torch.cuda.max_memory_allocated())
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return peak
    except ImportError:
        return 0

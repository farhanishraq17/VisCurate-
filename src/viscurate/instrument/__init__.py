"""A4 telemetry layer — cost/timing instrumentation wrapped around existing pipeline stages."""

from viscurate.instrument.telemetry import (
    Recorder,
    RunManifest,
    gpu_timer,
    peak_gpu_memory_bytes,
    recorder,
    timer,
)

__all__ = [
    "Recorder",
    "RunManifest",
    "gpu_timer",
    "peak_gpu_memory_bytes",
    "recorder",
    "timer",
]

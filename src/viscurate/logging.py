"""structlog-based JSON logging setup (CLAUDE.md Phase 0).

One call to :func:`configure_logging` installs a processor chain that renders either
machine-readable JSON (for runs/manifests) or a colorless key=value console format (for
local dev). Loggers are retrieved with :func:`get_logger`; we keep the return type as
``Any`` because structlog ships without complete stubs.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False


def configure_logging(level: str = "INFO", *, json: bool = True) -> None:
    """Configure structlog + stdlib logging. Idempotent across repeated calls."""
    global _CONFIGURED

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=numeric_level, force=True)

    renderer: Any = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None, **initial: Any) -> Any:
    """Return a bound structlog logger, configuring with defaults on first use."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name, **initial)

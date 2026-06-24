"""Query-derived usage statistics for the curation environment (CLAUDE.md Layer E)."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from viscurate.curation.state import UsageStats
from viscurate.downstream.query import QueryManifest

__all__ = ["UsageConfig", "usage_from_queries"]


class UsageConfig(BaseModel):
    """Deterministic synthetic usage-log knobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_count: int = Field(default=20, ge=1)
    zipf_alpha: float = Field(default=1.2, ge=0)


def usage_from_queries(
    manifest: QueryManifest,
    *,
    cfg: UsageConfig | None = None,
    registry_ids: Iterable[str] | None = None,
) -> UsageStats:
    """Build :class:`UsageStats` from query relevance plus a deterministic Zipfian count.

    ``referenced`` is the exact set of skills that any query expects. ``counts`` is a synthetic
    usage frequency over those referenced skills, rank-ordered by first appearance in the query
    stream. Unreferenced registry ids are omitted, which is equivalent to zero usage.
    """
    cfg = cfg or UsageConfig()
    counts: dict[str, int] = {}
    order: list[str] = []
    for q in manifest.entries:
        for sid in q.expected_skill_ids:
            if sid not in counts:
                counts[sid] = 0
                order.append(sid)
            counts[sid] += 1

    for rank, sid in enumerate(order, start=1):
        counts[sid] += max(1, int(round(cfg.base_count / (rank**cfg.zipf_alpha))))

    if registry_ids is not None:
        valid = set(registry_ids)
        counts = {sid: n for sid, n in counts.items() if sid in valid}
        referenced = frozenset(sid for sid in manifest.referenced_skill_ids() if sid in valid)
    else:
        referenced = manifest.referenced_skill_ids()

    return UsageStats(counts=counts, referenced=referenced)

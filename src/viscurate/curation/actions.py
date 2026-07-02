"""The curation action vocabulary and the structured action outcome (CLAUDE.md §3.2, §3.5.7).

The agent answers "what *should* be done?" by proposing one of eight actions; the environment
applies it (after a verifier gate for the structural ones) and returns an :class:`ActionResult`.
Both types are frozen, JSON-serializable Pydantic models — the per-episode action log is a
tuple of :class:`ActionResult`, scored against the ideal-action key in Phase 8.

Field conventions mirror :class:`viscurate.corruption.types.IdealAction` so the two line up
directly when scored: ``primary`` is the skill *acted on* (merged-away / folded-away / removed
/ modified), ``secondary`` is the surviving skill (merge target / generalizer), when applicable.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

__all__ = [
    "STRUCTURAL_ACTIONS",
    "Action",
    "ActionKind",
    "ActionResult",
    "ActionStatus",
]


class ActionKind(StrEnum):
    """The eight curation actions (CLAUDE.md §3.2)."""

    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"
    RETRIEVE = "retrieve"
    MERGE = "merge"
    SPLIT = "split"
    PARAMETERIZE = "parameterize"
    END = "end"


#: Structural edits that change functional coverage — they **cannot proceed without a
#: certifying relation from the verifier** (CLAUDE.md §3.2 gating).
STRUCTURAL_ACTIONS: frozenset[ActionKind] = frozenset(
    {ActionKind.MERGE, ActionKind.SPLIT, ActionKind.PARAMETERIZE}
)


class ActionStatus(StrEnum):
    """How an action resolved."""

    APPLIED = "applied"  # mutated the library
    REJECTED = "rejected"  # verifier gate denied the structural edit (no certifying relation)
    BLOCKED = "blocked"  # sandbox/trust gate: untrusted code cannot be verified or executed
    NOOP = "noop"  # nothing to mutate (e.g. retrieve / end)
    INVALID = "invalid"  # malformed action (unknown id, self-pair, missing field)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Action(_Frozen):
    """One proposed curation action — a flat union of optional fields keyed by ``kind``.

    Fields used per kind:

    * MERGE / PARAMETERIZE : ``primary`` (acted-on), ``secondary`` (survivor)
    * SPLIT                : ``primary`` (the skill to split) + ``new_skill_id`` / ``fn_source``
    * REMOVE               : ``primary``
    * MODIFY               : ``primary`` + any of ``new_name`` / ``new_description`` /
      (``param_name``, ``value``) — output-preserving repairs the agent can make without code
    * ADD                  : ``new_skill_id`` / ``new_name`` / ``new_description`` / ``family`` /
      optional ``fn_source`` (agent-authored code; untrusted and sandbox-gated)
    * RETRIEVE             : ``query`` (and/or ``primary``) — a no-op observation, logged
    * END                  : (no fields)

    ``rationale`` is the agent's stated reason, kept for the action-cost / Pareto analysis.
    """

    kind: ActionKind
    primary: str = ""
    secondary: str = ""
    rationale: str = ""

    # modify (output-preserving repairs) — same flat-union discipline as CorruptionEntry.
    new_name: str = ""
    new_description: str = ""
    param_name: str = ""
    value: bool | int | float | str | None = None

    # add (agent-authored skill: registered untrusted, never executed in v1)
    new_skill_id: str = ""
    family: str = ""
    fn_source: str = ""

    # retrieve
    query: str = ""

    @property
    def is_structural(self) -> bool:
        return self.kind in STRUCTURAL_ACTIONS


class ActionResult(_Frozen):
    """The structured outcome of applying one action — the canonical log record.

    For a rejected structural edit, ``relation`` / ``distances`` / ``alternatives`` carry the
    verifier's evidence so the rejection is *actionable* (CLAUDE.md §3.5.7) rather than a retry
    signal. ``reason`` is the human-readable summary (it quotes the deciding distances, e.g.
    ``"DISTINCT — worst-case L∞ 0.31, LPIPS 0.27"``).
    """

    action: Action
    status: ActionStatus
    reason: str = ""
    relation: str = ""  # the verifier's relation for a structural edit (empty otherwise)
    direction: str = ""
    distances: dict[str, float] = {}
    alternatives: tuple[str, ...] = ()
    size_before: int = 0
    size_after: int = 0

    @property
    def applied(self) -> bool:
        return self.status is ActionStatus.APPLIED

    def rejection_feedback(self) -> dict[str, object]:
        """The structured ``{"rejected": true, ...}`` payload the agent sees (CLAUDE.md §3.5.7)."""
        return {
            "rejected": self.status is ActionStatus.REJECTED,
            "relation": self.relation,
            "reason": self.reason,
            "alternatives": list(self.alternatives),
        }

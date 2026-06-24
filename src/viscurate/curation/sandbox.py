"""The execution-trust boundary for curation (CLAUDE.md §5, D6 — Phase-6 sandbox hardening).

Phase 6 is where **agent-generated skills first appear**. The project's non-negotiable rule
(CLAUDE.md §5) is that such code stays **blocked pending a hardened sandbox** — network
namespace, restricted filesystem, CPU/memory caps, a hard timeout, and no ``eval``/``exec`` of
skill code in the main process. That hardened sandbox is **security-sensitive and is left for
human review**; this module does not implement or enable untrusted execution. It makes the
boundary explicit and testable:

* :class:`ExecutionPolicy` decides whether a skill may be executed / output-verified at all.
  Untrusted skills are **blocked** with a structured, actionable reason.
* The curation environment consults this before feeding a skill to the in-process
  ``BatteryEvaluator`` (which has no trust gate of its own) or to a structural edit — so an
  agent-added skill can never have its ``fn`` run until the hardened sandbox is reviewed.

``allow_untrusted`` mirrors :class:`viscurate.config.ExecutorConfig.allow_untrusted` and MUST
stay ``False`` outside a reviewed, hardened sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass

from viscurate.skills.model import ComparatorView, Skill

__all__ = [
    "HARDENING_PLAN",
    "REVIEW_REQUIRED",
    "ExecutionDecision",
    "ExecutionPolicy",
]

#: The reason returned when an untrusted skill is blocked — surfaced to the agent and logged.
REVIEW_REQUIRED = (
    "BLOCKED: untrusted (agent-generated) skill cannot be executed or output-verified until "
    "the hardened sandbox is reviewed (CLAUDE.md §5)"
)

#: The deferred, human-review-gated controls that must be in place before ``allow_untrusted``
#: may be flipped. Documented here so the boundary is auditable, not implicit.
HARDENING_PLAN: tuple[str, ...] = (
    "run skill fn in a network namespace with no egress",
    "restricted/read-only filesystem (chroot or container), no host mounts",
    "hard CPU and memory rlimits + a wall-clock timeout (already in SandboxedExecutor on POSIX)",
    "no eval/exec of skill source in the main process; validate params against schema first",
    "human review sign-off recorded before allow_untrusted is set True",
)


@dataclass(frozen=True)
class ExecutionDecision:
    """Whether a skill may run (be executed / output-verified) under the current policy."""

    permitted: bool
    reason: str = ""


@dataclass(frozen=True)
class ExecutionPolicy:
    """Gate on which skills may be executed or output-verified (the hardened-sandbox boundary).

    With ``allow_untrusted=False`` (the only safe setting in v1) a ``trusted=False`` skill is
    blocked. Trusted built-ins pass.
    """

    allow_untrusted: bool = False

    def gate(self, *, trusted: bool) -> ExecutionDecision:
        if trusted or self.allow_untrusted:
            return ExecutionDecision(permitted=True)
        return ExecutionDecision(permitted=False, reason=REVIEW_REQUIRED)

    def gate_skill(self, skill: Skill) -> ExecutionDecision:
        return self.gate(trusted=skill.metadata.trusted)

    def gate_view(self, view: ComparatorView, *, trusted: bool) -> ExecutionDecision:
        """Gate by a comparator view + a separately-known trust flag (text-blind path)."""
        del view  # identity only; the trust flag is the gate
        return self.gate(trusted=trusted)

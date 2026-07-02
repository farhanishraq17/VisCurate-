"""The curation environment (CLAUDE.md §3.2, §3.5.7 — Phase 6).

The agent half of the verifier/agent split: a curatable library, the action API
(``add / remove / modify / retrieve / merge / split / parameterize / end``), the **hard
verifier gate** on structural edits with structured rejection-and-reason, the relation→action
map, usage-aware advisories, agent adapters (scripted + LLM behind a swappable client), and the
sandbox trust boundary that keeps agent-generated code blocked pending review.

The verifier/agent split is enforced structurally: the environment hands the verifier a
``ComparatorView`` + ``OutputProvider`` (never a description), and the only place a structural
edit can be applied is through :meth:`CurationEnvironment.apply`, which calls the gate first.
"""

from __future__ import annotations

from viscurate.curation.actions import (
    STRUCTURAL_ACTIONS,
    Action,
    ActionKind,
    ActionResult,
    ActionStatus,
)
from viscurate.curation.agent import (
    AnthropicClient,
    CurationAgent,
    LlmCurationAgent,
    OllamaClient,
    ScriptedAgent,
    list_ollama_models,
    parse_action,
)
from viscurate.curation.environment import CurationEnvironment, EpisodeResult, run_episode
from viscurate.curation.gating import GateDecision, gate_structural
from viscurate.curation.hardened import HardenedExecutor, HardenedRunResult
from viscurate.curation.sandbox import HARDENING_PLAN, REVIEW_REQUIRED, ExecutionPolicy
from viscurate.curation.state import CurationState, SkillSummary, UsageStats

__all__ = [
    "HARDENING_PLAN",
    "REVIEW_REQUIRED",
    "STRUCTURAL_ACTIONS",
    "Action",
    "ActionKind",
    "ActionResult",
    "ActionStatus",
    "AnthropicClient",
    "CurationAgent",
    "CurationEnvironment",
    "CurationState",
    "EpisodeResult",
    "ExecutionPolicy",
    "GateDecision",
    "HardenedExecutor",
    "HardenedRunResult",
    "LlmCurationAgent",
    "OllamaClient",
    "ScriptedAgent",
    "SkillSummary",
    "UsageStats",
    "gate_structural",
    "list_ollama_models",
    "parse_action",
    "run_episode",
]

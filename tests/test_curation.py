"""Phase 6 — the curation environment (CLAUDE.md §3.2, §3.5.7).

Deterministic and ML-free: the verifier is driven by the same fake backends as the Phase-3
suite, so every gate branch is exercised exactly. The exit criteria are asserted directly:
``merge(blur_gaussian, blur_box)`` is rejected with LPIPS in the rejection, an exact-duplicate
merge is approved, an untrusted (agent-added) skill is BLOCKED, ``end()`` is clean, and every
action is logged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from viscurate.baselines.judges import LlmUnavailableError, UnavailableLlmClient
from viscurate.config import ThresholdConfig
from viscurate.curation import (
    REVIEW_REQUIRED,
    Action,
    ActionKind,
    ActionStatus,
    CurationEnvironment,
    ExecutionPolicy,
    LlmCurationAgent,
    ScriptedAgent,
    UsageStats,
    parse_action,
    run_episode,
)
from viscurate.curation.agent import OllamaClient
from viscurate.equivalence.param_alignment import load_param_alignment
from viscurate.equivalence.relations import Relation
from viscurate.skills.library import build_builtin_registry
from viscurate.skills.library._build import make_skill
from viscurate.skills.model import Image, Params, Skill, SkillMetadata


class FakePerceptual:
    name = "fake-perc"

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return float("inf")
        return float(np.mean(np.abs(a - b)))


def _flip(image: Image, params: Params, seed: int) -> Image:
    return np.ascontiguousarray(image[:, ::-1])


def _add(delta: int):  # type: ignore[no-untyped-def]
    def fn(image: Image, params: Params, seed: int) -> Image:
        return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    return fn


def _mul(factor: float):  # type: ignore[no-untyped-def]
    def fn(image: Image, params: Params, seed: int) -> Image:
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return fn


def _skill(skill_id: str, fn, family: str = "test") -> Skill:  # type: ignore[no-untyped-def]
    return make_skill(skill_id, skill_id, "secret description", fn, family)


def _battery() -> list[tuple[str, Image]]:
    rng = np.random.default_rng(7)
    tex = rng.integers(20, 200, size=(48, 48, 3), dtype=np.uint8)
    grad = np.tile(np.linspace(20, 200, 48, dtype=np.uint8)[None, :, None], (48, 1, 3))
    return [("tex", tex), ("grad", grad)]


def _env(
    skills: Sequence[Skill],
    battery: list[tuple[str, Image]] | None = None,
    **kwargs: object,
) -> CurationEnvironment:
    return CurationEnvironment.from_skills(
        skills,
        battery if battery is not None else _battery(),
        thresholds=ThresholdConfig(),
        perceptual=FakePerceptual(),
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------------------------
# Exit criterion 1 — merge(blur_gaussian, blur_box) is rejected, with LPIPS in the rejection.
# --------------------------------------------------------------------------------------------


def test_merge_blur_gaussian_box_rejected_with_lpips() -> None:
    reg = build_builtin_registry()
    g, b = reg.get("blur_gaussian_v1"), reg.get("blur_box_v1")
    align = load_param_alignment("configs/param_alignment.yaml")
    env = _env([g, b], alignment=align)
    res = env.apply(
        Action(kind=ActionKind.MERGE, primary="blur_box_v1", secondary="blur_gaussian_v1")
    )
    assert res.status is ActionStatus.REJECTED
    assert res.relation not in (Relation.EXACT.value, Relation.PERCEPTUAL.value)
    assert "lpips" in res.distances  # the matched-sweep worst-case LPIPS is in the rejection
    assert res.alternatives  # actionable: keep_separate / parameterize
    assert len(env.registry) == 2  # nothing merged
    feedback = res.rejection_feedback()
    assert feedback["rejected"] is True


def test_merge_rejection_reason_quotes_lpips() -> None:
    # A genuinely DISTINCT, non-commuting pair → the reason literally quotes "LPIPS" (§3.2 format).
    addbig = _skill("addbig_v1", _add(80))
    mul3 = _skill("mul3_v1", _mul(3.0))
    env = _env([addbig, mul3])
    res = env.apply(Action(kind=ActionKind.MERGE, primary="addbig_v1", secondary="mul3_v1"))
    assert res.status is ActionStatus.REJECTED
    assert res.relation == Relation.DISTINCT.value
    assert "LPIPS" in res.reason


# --------------------------------------------------------------------------------------------
# Exit criterion 2 — an exact-duplicate merge is approved.
# --------------------------------------------------------------------------------------------


def test_exact_duplicate_merge_approved() -> None:
    flip = _skill("flip_v1", _flip, "geo")
    clone = _skill("flip_clone_v1", _flip, "geo")  # same fn, different id → EXACT
    env = _env([flip, clone])
    res = env.apply(Action(kind=ActionKind.MERGE, primary="flip_clone_v1", secondary="flip_v1"))
    assert res.status is ActionStatus.APPLIED
    assert res.relation == Relation.EXACT.value
    assert res.size_before == 2 and res.size_after == 1
    assert "flip_clone_v1" not in env.registry  # the duplicate was folded away
    assert "flip_v1" in env.registry  # the canonical survives


# --------------------------------------------------------------------------------------------
# Exit criterion 3 — an untrusted (agent-generated) skill is BLOCKED from verification/execution.
# --------------------------------------------------------------------------------------------


def test_added_skill_is_untrusted_and_merge_is_blocked() -> None:
    flip = _skill("flip_v1", _flip, "geo")
    env = _env([flip])
    added = env.apply(
        Action(kind=ActionKind.ADD, new_skill_id="agent_skill_v1", new_name="mystery")
    )
    assert added.status is ActionStatus.APPLIED
    new_skill = env.registry.get("agent_skill_v1")
    assert new_skill.metadata.trusted is False
    assert new_skill.metadata.provenance == "agent"

    blocked = env.apply(
        Action(kind=ActionKind.MERGE, primary="agent_skill_v1", secondary="flip_v1")
    )
    assert blocked.status is ActionStatus.BLOCKED
    assert REVIEW_REQUIRED in blocked.reason
    assert len(env.registry) == 2  # no merge happened


def test_execution_policy_blocks_untrusted_allows_trusted() -> None:
    policy = ExecutionPolicy()
    assert policy.gate(trusted=True).permitted is True
    denied = policy.gate(trusted=False)
    assert denied.permitted is False and denied.reason == REVIEW_REQUIRED
    # Only a reviewed, hardened sandbox flips this — exercised by the override.
    assert ExecutionPolicy(allow_untrusted=True).gate(trusted=False).permitted is True


def test_split_is_blocked_pending_hardened_sandbox() -> None:
    flip = _skill("flip_v1", _flip, "geo")
    env = _env([flip])
    res = env.apply(Action(kind=ActionKind.SPLIT, primary="flip_v1"))
    assert res.status is ActionStatus.BLOCKED


# --------------------------------------------------------------------------------------------
# Exit criterion 4 — end() is clean; exit criterion 5 — every action is logged.
# --------------------------------------------------------------------------------------------


def test_end_is_clean_and_actions_are_logged() -> None:
    flip = _skill("flip_v1", _flip, "geo")
    clone = _skill("flip_clone_v1", _flip, "geo")
    env = _env([flip, clone])
    agent = ScriptedAgent(
        [
            Action(kind=ActionKind.MERGE, primary="flip_clone_v1", secondary="flip_v1"),
            Action(kind=ActionKind.END),
        ]
    )
    episode = run_episode(env, agent)
    assert episode.ended is True
    assert episode.size_before == 2 and episode.size_after == 1
    assert episode.compression == 1
    assert episode.applied_kinds().get("merge") == 1
    # Every action is logged and the log is JSON-serializable (Phase-8 scoring input).
    assert len(episode.log) == 2
    assert episode.log[-1].action.kind is ActionKind.END
    assert episode.log[-1].status is ActionStatus.NOOP
    for record in episode.log:
        json.loads(record.model_dump_json())  # round-trips


def test_run_episode_respects_budget() -> None:
    flip = _skill("flip_v1", _flip, "geo")
    env = _env([flip], budget=2)
    agent = ScriptedAgent([Action(kind=ActionKind.RETRIEVE, query=f"q{i}") for i in range(5)])
    episode = run_episode(env, agent)
    assert episode.ended is False
    assert len(episode.log) == 2  # stopped at the budget, not the script
    assert env.budget_remaining == 0


# --------------------------------------------------------------------------------------------
# State carries NO internal ground-truth labels (CLAUDE.md §1.2).
# --------------------------------------------------------------------------------------------


def test_state_never_exposes_internal_labels() -> None:
    buggy = Skill(
        id="buggy_v1",
        name="buggy",
        description="d",
        fn=_flip,
        params_schema=make_skill("x", "x", "x", _flip, "f").params_schema,
        metadata=SkillMetadata(family="geo", is_buggy=True, is_dead=True),
    )
    env = _env([buggy])
    state = env.observe()
    summary = state.skills[0]
    assert not hasattr(summary, "is_buggy")
    assert not hasattr(summary, "is_dead")
    blob = state.model_dump_json()
    assert "is_buggy" not in blob and "is_dead" not in blob


# --------------------------------------------------------------------------------------------
# Relation → action map: parameterize gated by SUBSUMPTION direction.
# --------------------------------------------------------------------------------------------


def test_parameterize_approved_on_subsumption() -> None:
    reg = build_builtin_registry()
    r90, rc = reg.get("rotate_90_v1"), reg.get("rotate_canvas_degrees_v1")
    align = load_param_alignment("configs/param_alignment.yaml")
    env = _env([r90, rc], alignment=align)
    res = env.apply(
        Action(
            kind=ActionKind.PARAMETERIZE,
            primary="rotate_90_v1",  # the specialization (subsumed)
            secondary="rotate_canvas_degrees_v1",  # the generalizer
        )
    )
    assert res.status is ActionStatus.APPLIED
    assert res.relation == Relation.SUBSUMPTION.value
    assert "rotate_90_v1" not in env.registry  # specialization folded away
    assert "rotate_canvas_degrees_v1" in env.registry


def test_parameterize_rejected_on_wrong_direction() -> None:
    reg = build_builtin_registry()
    r90, rc = reg.get("rotate_90_v1"), reg.get("rotate_canvas_degrees_v1")
    align = load_param_alignment("configs/param_alignment.yaml")
    env = _env([r90, rc], alignment=align)
    # Folding the generalizer into the specialization is the wrong direction → rejected.
    res = env.apply(
        Action(
            kind=ActionKind.PARAMETERIZE,
            primary="rotate_canvas_degrees_v1",
            secondary="rotate_90_v1",
        )
    )
    assert res.status is ActionStatus.REJECTED
    assert len(env.registry) == 2


# --------------------------------------------------------------------------------------------
# remove / modify / retrieve, usage advisories.
# --------------------------------------------------------------------------------------------


def test_remove_flags_referenced_skill() -> None:
    flip = _skill("flip_v1", _flip, "geo")
    usage = UsageStats(counts={"flip_v1": 5}, referenced=frozenset({"flip_v1"}))
    env = _env([flip], usage=usage)
    res = env.apply(Action(kind=ActionKind.REMOVE, primary="flip_v1"))
    assert res.status is ActionStatus.APPLIED
    assert "WARNING" in res.reason and "flip_v1" not in env.registry


def test_modify_metadata_is_output_preserving() -> None:
    flip = _skill("flip_v1", _flip, "geo")
    env = _env([flip])
    res = env.apply(
        Action(kind=ActionKind.MODIFY, primary="flip_v1", new_description="accurate description")
    )
    assert res.status is ActionStatus.APPLIED
    skill = env.registry.get("flip_v1")
    assert skill.description == "accurate description"
    assert skill.metadata.trusted is True  # metadata repair keeps it trusted


def test_retrieve_and_unknown_ids_do_not_mutate() -> None:
    flip = _skill("flip_v1", _flip, "geo")
    env = _env([flip])
    assert (
        env.apply(Action(kind=ActionKind.RETRIEVE, query="make it gray")).status
        is ActionStatus.NOOP
    )
    assert (
        env.apply(Action(kind=ActionKind.REMOVE, primary="nope_v1")).status is ActionStatus.INVALID
    )
    assert len(env.registry) == 1


def test_parameterize_usage_advisory_when_heavily_used() -> None:
    reg = build_builtin_registry()
    r90, rc = reg.get("rotate_90_v1"), reg.get("rotate_canvas_degrees_v1")
    align = load_param_alignment("configs/param_alignment.yaml")
    usage = UsageStats(counts={"rotate_90_v1": 9})
    env = _env([r90, rc], alignment=align, usage=usage)
    res = env.apply(
        Action(
            kind=ActionKind.PARAMETERIZE,
            primary="rotate_90_v1",
            secondary="rotate_canvas_degrees_v1",
        )
    )
    # The verifier still permits it (a fact), but the result surfaces the usage cost for the agent.
    assert res.status is ActionStatus.APPLIED
    assert "usage 9" in res.reason


# --------------------------------------------------------------------------------------------
# Agent adapters.
# --------------------------------------------------------------------------------------------


def test_parse_action_extracts_json() -> None:
    reply = 'Sure, here is my action:\n{"kind": "merge", "primary": "a_v1", "secondary": "b_v1"}'
    action = parse_action(reply)
    assert action.kind is ActionKind.MERGE
    assert action.primary == "a_v1" and action.secondary == "b_v1"


def test_llm_agent_unavailable_raises_rather_than_fabricates() -> None:
    agent = LlmCurationAgent(UnavailableLlmClient())
    assert agent.available is False
    env = _env([_skill("flip_v1", _flip, "geo")])
    try:
        agent.propose(env.observe())
    except LlmUnavailableError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected LlmUnavailableError")


def test_llm_agent_parses_a_returned_action() -> None:
    class FakeClient:
        name = "fake"

        def complete(self, prompt: str) -> str:
            return '{"kind": "end", "rationale": "library looks clean"}'

    agent = LlmCurationAgent(FakeClient())
    flip = _skill("flip_v1", _flip, "geo")
    env = _env([flip])
    action = agent.propose(env.observe())
    assert action.kind is ActionKind.END


def test_llm_agent_ends_on_unparseable_reply() -> None:
    class BadClient:
        name = "bad"

        def complete(self, prompt: str) -> str:
            return "I cannot help with that."

    agent = LlmCurationAgent(BadClient())
    flip = _skill("flip_v1", _flip, "geo")
    env = _env([flip])
    action = agent.propose(env.observe())
    assert action.kind is ActionKind.END  # terminates cleanly, never crashes


def test_ollama_client_name_and_no_network_at_construction() -> None:
    client = OllamaClient("llama3", host="http://localhost:11434/")
    assert client.name == "ollama:llama3"
    assert client.host == "http://localhost:11434"  # trailing slash trimmed

"""Curation agents — the policy that proposes actions (CLAUDE.md §3.2, D7).

A :class:`CurationAgent` maps a :class:`CurationState` to the next :class:`Action`. Two are
provided:

* :class:`ScriptedAgent` — replays a fixed action list (deterministic; the substrate for tests
  and for the ``no-curation`` / ``accumulate-only`` baselines of CLAUDE.md §3.4).
* :class:`LlmCurationAgent` — an LLM behind the swappable :class:`LlmClient` text-completion
  protocol (the same one the Phase-4 LLM judge uses). It renders the state, asks for one JSON
  action, and parses it. With no client configured it raises rather than fabricating an action
  (CLAUDE.md §5).

Two concrete clients ship: :class:`OllamaClient` (local, multi-model, dependency-free — stdlib
HTTP) and :class:`AnthropicClient` (Claude API, optional ``anthropic`` dependency). The agent is
**model-agnostic**: any object with ``name`` + ``complete(prompt) -> str`` plugs in (CLAUDE.md D7
"try several and compare").
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from viscurate.baselines.judges import LlmClient, LlmUnavailableError, UnavailableLlmClient
from viscurate.curation.actions import Action, ActionKind
from viscurate.curation.state import CurationState

__all__ = [
    "AnthropicClient",
    "CurationAgent",
    "LlmClient",
    "LlmCurationAgent",
    "OllamaClient",
    "ScriptedAgent",
    "list_ollama_models",
    "parse_action",
]


@runtime_checkable
class CurationAgent(Protocol):
    """Proposes the next curation action from the observed state."""

    def propose(self, state: CurationState) -> Action: ...


class ScriptedAgent:
    """Replays a fixed sequence of actions, then ends (deterministic — for tests/baselines)."""

    def __init__(self, actions: Sequence[Action]) -> None:
        self._actions = list(actions)
        self._i = 0

    def propose(self, state: CurationState) -> Action:
        del state  # scripted: ignores observations by design
        if self._i >= len(self._actions):
            return Action(kind=ActionKind.END, rationale="script exhausted")
        action = self._actions[self._i]
        self._i += 1
        return action


# --------------------------------------------------------------------------------------------
# LLM-driven agent.
# --------------------------------------------------------------------------------------------

_PROMPT = """\
You are curating a library of image-processing skills. Propose ONE action that improves the \
library (removes redundancy/defects, raises quality) without losing functional coverage.

Structural edits are GATED by an output-grounded verifier and only proceed if it certifies the \
relation: `merge` needs the pair to be EXACT or PERCEPTUAL duplicates; `parameterize` needs the \
primary to be a special case of (subsumed by) the secondary, or a semantic-preserving variant. \
The verifier compares OUTPUTS, not descriptions — do not assume two similarly-named skills are \
mergeable. Consult `used=` before folding/removing: do not remove or fold away a skill that is \
used or referenced unless it is a confirmed duplicate of a surviving skill.

Actions (JSON `kind` field):
- "merge": fold `primary` into the surviving `secondary` (exact/perceptual duplicate)
- "parameterize": fold the specialization `primary` into the generalizer `secondary`
- "remove": drop `primary` (a dead or broken skill with no surviving equivalent)
- "modify": fix `primary`'s `new_name`/`new_description` or a `param_name`+`value` default
- "retrieve": observe (no change) — `query` text
- "add": propose a new skill (`new_skill_id`,`new_name`,`new_description`) — untrusted, blocked
- "end": stop (the library is clean / no beneficial action remains)

Respond with EXACTLY ONE JSON object and nothing else, e.g.:
{{"kind": "merge", "primary": "blur_box_clone_v1", "secondary": "blur_box_v1"}}

Current state:
{state}

Your single JSON action:"""


class ActionParseError(ValueError):
    """Raised when an LLM reply cannot be parsed into an :class:`Action`."""


def parse_action(reply: str) -> Action:
    """Parse the first JSON object in ``reply`` into an :class:`Action` (raises on failure)."""
    start = reply.find("{")
    end = reply.rfind("}")
    if start < 0 or end <= start:
        raise ActionParseError(f"no JSON object in reply: {reply[:120]!r}")
    try:
        data = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"invalid JSON action: {exc}") from exc
    if not isinstance(data, dict) or "kind" not in data:
        raise ActionParseError(f"action JSON missing 'kind': {data!r}")
    try:
        return Action.model_validate(data)
    except Exception as exc:  # pydantic ValidationError → uniform parse error
        raise ActionParseError(f"action failed validation: {exc}") from exc


class LlmCurationAgent:
    """Drives an :class:`LlmClient`: render state → ask for one JSON action → parse it.

    On an unparseable / invalid reply, proposes ``end`` with the reason recorded (so an episode
    terminates cleanly rather than crashing). With the default unavailable client, ``propose``
    raises :class:`LlmUnavailableError` — the track is *not run* rather than fabricated.
    """

    def __init__(self, client: LlmClient | None = None) -> None:
        self.client: LlmClient = client or UnavailableLlmClient()

    @property
    def name(self) -> str:
        return f"llm-agent:{self.client.name}"

    @property
    def available(self) -> bool:
        return not isinstance(self.client, UnavailableLlmClient)

    def propose(self, state: CurationState) -> Action:
        reply = self.client.complete(_PROMPT.format(state=state.render()))
        try:
            return parse_action(reply)
        except ActionParseError as exc:
            return Action(kind=ActionKind.END, rationale=f"unparseable LLM action: {exc}")


# --------------------------------------------------------------------------------------------
# Concrete LLM clients (Ollama multi-model; Claude API optional).
# --------------------------------------------------------------------------------------------


class OllamaClient:
    """A local Ollama text-completion client (multi-model, dependency-free via stdlib HTTP).

    Talks to Ollama's stable ``POST /api/generate`` endpoint with ``stream=false``. No external
    dependency and no network at import time — the call happens only in :meth:`complete`.
    """

    def __init__(
        self, model: str, *, host: str = "http://localhost:11434", timeout: float = 120.0
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def complete(self, prompt: str) -> str:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise LlmUnavailableError(f"Ollama request failed ({self.host}): {exc}") from exc
        return str(body.get("response", ""))


def list_ollama_models(*, host: str = "http://localhost:11434", timeout: float = 10.0) -> list[str]:
    """Enumerate locally-installed Ollama models via ``GET /api/tags`` (CLAUDE.md D7)."""
    req = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        raise LlmUnavailableError(f"Ollama not reachable at {host}: {exc}") from exc
    return [str(m["name"]) for m in body.get("models", []) if "name" in m]


class AnthropicClient:
    """A Claude API text-completion client (optional ``anthropic`` dependency, imported lazily).

    Defaults to Claude Opus 4.8 with adaptive thinking — the current recommended model/params
    (verified against the Anthropic SDK reference). The API key is read from ``ANTHROPIC_API_KEY``
    by the SDK; it is never passed in source (CLAUDE.md §5 / org policy).
    """

    def __init__(self, *, model: str = "claude-opus-4-8", max_tokens: int = 4096) -> None:
        try:
            import anthropic  # lazy: the [agent] extra
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise LlmUnavailableError(
                "anthropic SDK not installed; `pip install viscurate[agent]` to use the Claude API"
            ) from exc
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"
        self._client: Any = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")

"""Judge tracks and the unified per-pair verdict (CLAUDE.md §3.5.9).

A **track** is one judge's predictions over the candidate pairs — the output-grounded verifier
or a text baseline. Every track produces, per pair, a :class:`Verdict` that exposes the same
two comparable fields: a fine ``relation`` guess and the binary ``mergeable`` decision. Scoring
both on the same axes is what lets the divergence table compare a text judge to the output
judge directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from viscurate.equivalence.relations import Direction, Relation

__all__ = ["Track", "Verdict"]


@dataclass(frozen=True)
class Verdict:
    """One judge's prediction for one pair, normalized across tracks.

    ``mergeable`` (predicts EXACT/PERCEPTUAL) is the binary axis compared in the divergence
    table; ``relation`` is the fine guess (full 6-way for the output verifier, coarse for text
    judges); ``uncertain`` flags the output verifier's abstention band; ``score`` is the
    deciding distance (output: lower = closer) or similarity (text: higher = closer), recorded
    for calibration/ROC with its meaning given by the track.
    """

    relation: Relation
    mergeable: bool
    direction: Direction = Direction.NONE
    score: float = float("nan")
    uncertain: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Track:
    """A named judge's predictions over the scored pairs.

    ``kind`` is ``"output"`` (text-blind verifier) or ``"text"`` (a baseline). ``ran`` is False
    when a track could not run (e.g. the LLM track with no client) so the report says
    *not run* rather than scoring fabricated verdicts.
    """

    name: str
    kind: str
    predictions: dict[tuple[str, str], Verdict] = field(default_factory=dict)
    ran: bool = True
    note: str = ""

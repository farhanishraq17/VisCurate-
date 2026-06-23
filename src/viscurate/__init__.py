"""VisCurate — output-grounded equivalence verification and library curation.

The package is layered so that the *output-grounded* path never imports anything that
touches a skill's text ``description`` (CLAUDE.md §1.2), and so that the skill harness
(Phases 0–1) is importable with zero ML dependencies (the comparators in Phase 3 live
behind the optional ``[ml]`` extra).
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]

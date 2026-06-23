"""Probe-battery manifest models (CLAUDE.md §2.1, §2.5).

The manifest is the small, tracked, authoritative record of the battery — images themselves
are large and regenerable (gitignored). Two project rules are enforced here:

* **No ``license=unknown``.** Every probe carries a concrete :class:`License`; the empty or
  "unknown" name is rejected at construction.
* **Coverage is checkable.** :meth:`ProbeManifest.assert_coverage` fails if a required domain
  or channel/format is missing or under-represented — a defect is only detectable if the
  battery exercises it.

The manifest also records the generator version, root seed, and canonicalization version so a
battery is fully reproducible (§2.5).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "CHANNEL_FORMATS",
    "DOMAINS",
    "ChannelFormat",
    "Domain",
    "License",
    "ProbeEntry",
    "ProbeManifest",
]

# Diversity axes (CLAUDE.md §2.1). A probe's domain says *what* it depicts; channel_format
# says *how* it is stored — the axis where domain-scoped bugs (Type 6) hide.
Domain = Literal[
    "natural", "document", "texture", "gradient", "shape", "noise", "colorchart", "degenerate"
]
ChannelFormat = Literal["rgb", "rgba", "gray", "gray16", "palette"]

DOMAINS: tuple[Domain, ...] = (
    "natural",
    "document",
    "texture",
    "gradient",
    "shape",
    "noise",
    "colorchart",
    "degenerate",
)
CHANNEL_FORMATS: tuple[ChannelFormat, ...] = ("rgb", "rgba", "gray", "gray16", "palette")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class License(BaseModel):
    """A concrete, redistributable license. ``unknown``/empty names are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    spdx: str = ""  # e.g. "CC0-1.0", "CC-BY-2.0"
    url: str = ""
    redistributable: bool = True
    allows_derivatives: bool = True

    @field_validator("name")
    @classmethod
    def _name_known(cls, v: str) -> str:
        if not v.strip() or v.strip().lower() == "unknown":
            raise ValueError("license name must be concrete (never empty/'unknown')")
        return v

    @model_validator(mode="after")
    def _usable(self) -> License:
        # The skills produce derivatives and the battery is redistributed; both must hold.
        if not (self.redistributable and self.allows_derivatives):
            raise ValueError(f"license {self.name!r} is not redistributable-with-derivatives")
        return self


# Convenience constants used by the generators / loaders.
CC0 = License(
    name="CC0 1.0 (public-domain dedication)",
    spdx="CC0-1.0",
    url="https://creativecommons.org/publicdomain/zero/1.0/",
)


class ProbeEntry(_Frozen):
    """One probe image's manifest record."""

    probe_id: str
    sha256: str
    domain: Domain
    channel_format: ChannelFormat
    height: int
    width: int
    source: str  # "synthetic" | "coco-test2017" | ...
    license: License
    attribution: str = ""
    notes: str = ""

    @field_validator("sha256")
    @classmethod
    def _hex64(cls, v: str) -> str:
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v.lower()):
            raise ValueError("sha256 must be 64 hex chars")
        return v.lower()


class ProbeManifest(_Frozen):
    """The versioned battery manifest (§2.5)."""

    manifest_version: str = "1"
    generator_version: str
    canon_version: str
    seed: int
    entries: tuple[ProbeEntry, ...]

    @model_validator(mode="after")
    def _unique_ids(self) -> ProbeManifest:
        ids = [e.probe_id for e in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("probe_id values must be unique")
        return self

    def domain_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {str(d): 0 for d in DOMAINS}
        for e in self.entries:
            counts[e.domain] += 1
        return counts

    def format_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {str(f): 0 for f in CHANNEL_FORMATS}
        for e in self.entries:
            counts[e.channel_format] += 1
        return counts

    def assert_coverage(
        self,
        *,
        required_domains: dict[str, int],
        required_formats: tuple[ChannelFormat, ...] = CHANNEL_FORMATS,
        required_degenerate: tuple[str, ...] = (),
    ) -> None:
        """Raise if a required domain/format/degenerate case is missing or under-represented.

        ``required_domains`` maps each domain that *must* be present to its minimum count, so
        an optional domain (e.g. ``natural`` when the network is offline) is simply omitted
        rather than silently weakening the floor on the domains that are present.
        """
        dc = self.domain_counts()
        thin = [f"{d}({dc[d]}<{n})" for d, n in required_domains.items() if dc[d] < n]
        if thin:
            raise ValueError(f"under-represented domains: {thin} (counts={dc})")
        fc = self.format_counts()
        missing_fmt = [f for f in required_formats if fc[f] == 0]
        if missing_fmt:
            raise ValueError(f"missing channel formats: {missing_fmt} (counts={fc})")
        present_notes = {e.notes for e in self.entries if e.domain == "degenerate"}
        missing_deg = [d for d in required_degenerate if d not in present_notes]
        if missing_deg:
            raise ValueError(f"missing degenerate cases: {missing_deg}")

    def __len__(self) -> int:
        return len(self.entries)

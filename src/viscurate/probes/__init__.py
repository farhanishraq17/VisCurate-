"""The probe battery and frozen reference oracle (CLAUDE.md Phase 2 / §2.1).

A *probe* is a single license-clean, content-hashed input image. The battery ``P`` is a
versioned, reproducible collection that spans the diversity axes a defect needs in order to
be detectable (domain, channel/format, signal, degenerate cases). The reference *oracle*
freezes every clean ``L0`` skill's output over ``P`` so later corruption can be *proven* to
have taken effect — the oracle is never used to assign relation labels (§2.1).
"""

from __future__ import annotations

from viscurate.probes.manifest import License, ProbeEntry, ProbeManifest

__all__ = ["License", "ProbeEntry", "ProbeManifest"]

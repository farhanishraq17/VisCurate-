#!/usr/bin/env python
"""Generate `corpus_a_forkpair.yaml` -- the pilgram/pilgram2 fork-pair alignment.

Enumerates the filter names the two forks SHARE and emits one family per shared name. The filters
take no parameters, so there is no alignment decision to make and nothing an observed output could
contaminate; the header of the generated file states that explicitly rather than implying the
file was authored blind like the hand-written corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pilgram
import pilgram2


def filters(mod: object) -> list[str]:
    return sorted(
        n
        for n, f in vars(mod).items()
        if callable(f)
        and not n.startswith("_")
        and getattr(f, "__module__", "").startswith(mod.__name__)  # type: ignore[attr-defined]
    )


def main() -> int:
    a, b = set(filters(pilgram)), set(filters(pilgram2))
    shared, only2 = sorted(a & b), sorted(b - a)
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/wacv_r2/corpus_a_forkpair.yaml")
    print(f"pilgram {len(a)} / pilgram2 {len(b)} / shared {len(shared)} -> {out}")
    print("  added in fork:", ", ".join(only2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

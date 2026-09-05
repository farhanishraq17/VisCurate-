#!/usr/bin/env python
"""A1 Corpus A, step 1 — parse the published cross-library mapping tables into a frozen pair list.

Reads the HTML snapshots captured under ``--snapshots`` (never the live pages, so the frozen list
is reproducible) and emits one row per published mapping row.

WHAT THESE ROWS ARE. The pages are *migration* tables: a named counterpart means the other library
has a built-in API for the same broad operation. They are category-level judgments, not assertions
of output equivalence at aligned parameters (A1/01 §1.1). The label taxonomy is preserved exactly
so ``direct`` / ``partial`` / ``none`` are never collapsed into one "claimed equivalent" bucket:

    direct   exactly one named counterpart
    partial  more than one named counterpart (a composition), or an explicit "(partial)" marker
    none     an em/en dash — no built-in counterpart is named, so the row names no second function
             and CANNOT be turned into a pair (A1/01 §6, finding 2)

Output is written with a sha256 over the canonical row serialization; that hash is what gets
committed before any execution happens.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

DASHES = {"-", "–", "—", "‒", "―", "", "n/a", "none"}

_TABLE_RE = re.compile(r"<table\b.*?</table>", re.I | re.S)
_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.I | re.S)
_CELL_RE = re.compile(r"<t[dh]\b.*?</t[dh]>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _text(cell_html: str) -> str:
    """Strip markup from one cell, keeping ``<br>``-separated names as ' | '-joined items."""
    s = re.sub(r"<br\s*/?>", " | ", cell_html, flags=re.I)
    s = _TAG_RE.sub(" ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _split_counterparts(cell: str) -> tuple[list[str], bool]:
    """Split a counterpart cell into names; the flag reports an explicit '(partial)' marker."""
    partial_marker = "partial" in cell.lower()
    cleaned = re.sub(r"\(\s*partial\s*\)", "", cell, flags=re.I)
    parts: list[str] = []
    for chunk in re.split(r"\||,|/|\bor\b|\band\b|\+", cleaned):
        name = chunk.strip().strip(".")
        if name and name.lower() not in DASHES:
            parts.append(name)
    return parts, partial_marker


def parse_snapshot(path: Path, source_url: str) -> list[dict[str, object]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, object]] = []
    for table_html in _TABLE_RE.findall(raw):
        trs = _ROW_RE.findall(table_html)
        if not trs:
            continue
        header = [_text(c) for c in _CELL_RE.findall(trs[0])]
        if len(header) < 2:
            continue
        lib_a, lib_b = header[0], header[1]
        for tr in trs[1:]:
            cells = [_text(c) for c in _CELL_RE.findall(tr)]
            if len(cells) < 2 or not cells[0]:
                continue
            counterparts, partial_marker = _split_counterparts(cells[1])
            if not counterparts:
                label = "none"
            elif partial_marker or len(counterparts) > 1:
                label = "partial"
            else:
                label = "direct"
            rows.append(
                {
                    "source_file": path.name,
                    "source_url": source_url,
                    "table_left_header": lib_a,
                    "table_right_header": lib_b,
                    "left_name": cells[0],
                    "right_cell_raw": cells[1],
                    "right_names": counterparts,
                    "published_label": label,
                    "extra_cells": cells[2:],
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", default="results/wacv_r2/corpusA/snapshots")
    ap.add_argument("--out", default="results/wacv_r2/corpusA")
    ap.add_argument("--retrieved", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    args = ap.parse_args()

    urls = {
        "albumentations-vs-torchvision.html": (
            "https://albumentations.ai/docs/albumentations-vs-torchvision/transforms/"
        ),
        "albumentations-vs-kornia.html": (
            "https://albumentations.ai/docs/albumentations-vs-kornia/transforms/"
        ),
    }
    snap_dir = Path(args.snapshots)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    provenance = []
    for name, url in urls.items():
        path = snap_dir / name
        if not path.exists():
            raise SystemExit(f"missing snapshot: {path}")
        parsed = parse_snapshot(path, url)
        rows.extend(parsed)
        provenance.append(
            {
                "snapshot": name,
                "url": url,
                "retrieved": args.retrieved,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows_parsed": len(parsed),
            }
        )

    # Canonical serialization -> the freeze hash. Sorted so the hash is order-independent.
    canonical = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    by_label: dict[str, int] = {}
    for r in rows:
        by_label[str(r["published_label"])] = by_label.get(str(r["published_label"]), 0) + 1

    payload = {
        "artifact": "corpus_a_published_rows",
        "frozen_sha256": digest,
        "n_rows": len(rows),
        "rows_by_published_label": by_label,
        "provenance": provenance,
        "label_semantics": {
            "direct": "exactly one named counterpart in the published table",
            "partial": "more than one named counterpart, or an explicit '(partial)' marker",
            "none": "an em/en dash: no built-in counterpart named, so the row is NOT a pair",
        },
        "rows": rows,
    }
    (out_dir / "published_rows.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "published_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_file", "left_lib", "right_lib", "left_name", "right_names", "label"])
        for r in rows:
            w.writerow(
                [
                    r["source_file"],
                    r["table_left_header"],
                    r["table_right_header"],
                    r["left_name"],
                    ";".join(r["right_names"]),  # type: ignore[arg-type]
                    r["published_label"],
                ]
            )
    print(f"parsed {len(rows)} published rows -> {out_dir}/published_rows.json")
    print(f"  by label: {by_label}")
    print(f"  frozen sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

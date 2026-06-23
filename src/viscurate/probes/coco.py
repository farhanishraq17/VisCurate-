"""License-clean natural-photo loader from COCO (CLAUDE.md §2.1).

COCO *annotations* are CC BY 4.0, but each image carries its own Flickr license (the COCO
``license`` field, ids 1–8). Because our skills **transform** images (produce derivatives)
and the battery is redistributed, we keep only the permissive, derivative-allowing,
redistributable licenses — **CC BY 2.0 (id 4)**, *no known copyright restrictions* (7), and
*US Government work* (8) — and exclude the NonCommercial (1–3), ShareAlike (5), and crucially
the **NoDerivs (6)** images. Each kept probe records its license, COCO source URL, and sha256.

Image metadata comes from the lightweight ``image_info_test2017.json`` (~1 MB); the metadata
is cached so repeat builds do not re-download. The license table below was verified against
the live COCO metadata, not recalled from memory.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from viscurate.probes.manifest import License
from viscurate.rng import SeedManager
from viscurate.skills.model import Image as Array

__all__ = [
    "PERMISSIVE_LICENSE_IDS",
    "FetchedProbe",
    "coco_license_table",
    "filter_permissive",
    "select_natural_probes",
]

_METADATA_URL = "http://images.cocodataset.org/annotations/image_info_test2017.zip"
_METADATA_MEMBER = "annotations/image_info_test2017.json"

# Permissive, redistributable, derivative-allowing licenses only.
PERMISSIVE_LICENSE_IDS: tuple[int, ...] = (4, 7, 8)


def coco_license_table() -> dict[int, License]:
    """COCO license id → :class:`License` for the permissive subset we accept."""
    return {
        4: License(
            name="Attribution License (CC BY 2.0)",
            spdx="CC-BY-2.0",
            url="http://creativecommons.org/licenses/by/2.0/",
        ),
        7: License(
            name="No known copyright restrictions",
            url="http://flickr.com/commons/usage/",
        ),
        8: License(
            name="United States Government Work",
            url="http://www.usa.gov/copyright.shtml",
        ),
    }


@dataclass(frozen=True)
class FetchedProbe:
    base_id: str
    array: Array
    license: License
    source: str
    attribution: str
    notes: str = ""


def _cached_metadata(cache_dir: Path, timeout: float) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "coco_image_info_test2017.json"
    if cached.exists():
        data = json.loads(cached.read_text(encoding="utf-8"))
    else:
        raw = urllib.request.urlopen(_METADATA_URL, timeout=timeout).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            meta = json.loads(zf.read(_METADATA_MEMBER))
        cached.write_text(json.dumps(meta), encoding="utf-8")
        data = meta
    images = data["images"]
    assert isinstance(images, list)
    return images


def filter_permissive(
    images: list[dict[str, Any]], permissive_ids: tuple[int, ...]
) -> list[dict[str, Any]]:
    """Keep only images whose COCO license id is in ``permissive_ids``."""
    return [im for im in images if int(im.get("license", -1)) in permissive_ids]


def _download_rgb(url: str, timeout: float) -> Array:
    raw = urllib.request.urlopen(url, timeout=timeout).read()
    with PILImage.open(io.BytesIO(raw)) as im:
        rgb = im.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def select_natural_probes(
    sm: SeedManager,
    n: int,
    cache_dir: Path,
    *,
    permissive_ids: tuple[int, ...] = PERMISSIVE_LICENSE_IDS,
    timeout: float = 30.0,
    candidate_factor: int = 4,
) -> list[FetchedProbe]:
    """Fetch up to ``n`` permissively-licensed natural photos, deterministically chosen.

    Selection is seeded (reproducible). Images that fail to download are skipped; we draw from
    a candidate pool ``candidate_factor`` × larger so transient failures still yield ``n``.
    """
    table = coco_license_table()
    images = filter_permissive(_cached_metadata(cache_dir, timeout), permissive_ids)
    order = sm.generator("coco", "select").permutation(len(images))
    pool = [images[i] for i in order[: n * candidate_factor]]

    out: list[FetchedProbe] = []
    for im in pool:
        if len(out) >= n:
            break
        lic_id = int(im["license"])
        coco_url = str(im["coco_url"])
        image_id = int(im["id"])
        try:
            arr = _download_rgb(coco_url, timeout)
        except Exception:
            continue
        lic = table[lic_id]
        out.append(
            FetchedProbe(
                base_id=f"coco_{image_id:012d}",
                array=arr,
                license=lic,
                source="coco-test2017",
                attribution=f"COCO test2017 #{image_id}; {lic.name}; source {coco_url}",
                notes="natural photo",
            )
        )
    return out

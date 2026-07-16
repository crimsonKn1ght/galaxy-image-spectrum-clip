"""Verify the HATS catalogs named in a crossmatch config before running a real Phase 0 build.

This automates the three build-time checks from the plan, all cheaply (catalog metadata plus a
one-row peek, no full download):

1. that the configured ``hats_path`` for each catalog actually opens;
2. that the configured column names (image / flux / wavelength / redshift / ra / dec) are present;
3. whether the catalog carries the pixel/flux arrays or is coordinates-only.

It reads the same config keys that ``crossmatch/lsdb_match.py`` reads, so a PASS here means the build
will find what it expects. Torch-free: only needs ``lsdb`` (and pyyaml). Exit code is non-zero if any
configured column is missing or a catalog fails to open, so it is CI/script friendly.

Usage:
    python scripts/verify_catalogs.py --config configs/crossmatch_legacy_desi.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Tuple

import yaml


def _looks_like_array(cell: Any) -> bool:
    """True if a cell holds a multi-element array (an image/spectrum), not a scalar coordinate."""
    import numpy as np

    if isinstance(cell, dict) and "array" in cell:
        cell = cell["array"]
    try:
        arr = np.asarray(cell)
    except Exception:
        return False
    return arr.ndim >= 1 and arr.size > 1


def _check_catalog(
    name: str, hats_path: str, required_columns: List[str], array_columns: List[str], head_rows: int
) -> Tuple[bool, List[str]]:
    """Open one catalog and report column presence + arrays-present. Returns (ok, messages)."""
    messages: List[str] = [f"[{name}] hats_path: {hats_path}"]

    if hats_path.startswith("VERIFY"):
        messages.append(
            f"[{name}] FAIL: hats_path is still a placeholder. Set the real HATS path from the "
            "UniverseTBD/multimodal-universe-hats collection."
        )
        return False, messages

    import lsdb

    try:
        catalog = lsdb.open_catalog(hats_path)
    except Exception as exc:  # noqa: BLE001 - surface any open failure to the user
        messages.append(f"[{name}] FAIL: could not open catalog: {exc}")
        return False, messages

    available = list(catalog.columns)
    messages.append(f"[{name}] columns ({len(available)}): {available}")

    ok = True
    missing = [c for c in required_columns if c not in available]
    if missing:
        ok = False
        messages.append(f"[{name}] FAIL: missing configured columns: {missing}")
    else:
        messages.append(f"[{name}] OK: all configured columns present.")

    try:
        head = catalog.head(head_rows)
    except Exception as exc:  # noqa: BLE001
        messages.append(f"[{name}] WARN: could not peek rows (arrays-present check skipped): {exc}")
        return ok, messages

    if len(head) == 0:
        messages.append(f"[{name}] WARN: catalog head is empty; cannot check arrays-present.")
        return ok, messages

    row = head.iloc[0]
    for column in array_columns:
        if column not in available:
            continue
        if _looks_like_array(row[column]):
            messages.append(f"[{name}] OK: column '{column}' holds an array.")
        else:
            ok = False
            messages.append(
                f"[{name}] FAIL: column '{column}' is not a multi-element array - this catalog may be "
                "coordinates-only, so arrays must be joined from the base MultimodalUniverse/* dataset."
            )
    return ok, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify HATS catalogs before a Phase 0 build")
    parser.add_argument("--config", type=str, default="configs/crossmatch_legacy_desi.yaml")
    parser.add_argument("--head-rows", type=int, default=1, help="rows to peek for the arrays check")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    img = config["image_catalog"]
    spec = config["spectrum_catalog"]

    image_ok, image_messages = _check_catalog(
        name="image",
        hats_path=img["hats_path"],
        required_columns=[img["ra_column"], img["dec_column"], img["image_column"]],
        array_columns=[img["image_column"]],
        head_rows=args.head_rows,
    )
    spectrum_ok, spectrum_messages = _check_catalog(
        name="spectrum",
        hats_path=spec["hats_path"],
        required_columns=[
            spec["ra_column"],
            spec["dec_column"],
            spec["flux_column"],
            spec["wavelength_column"],
            spec["redshift_column"],
        ],
        array_columns=[spec["flux_column"], spec["wavelength_column"]],
        head_rows=args.head_rows,
    )

    for line in image_messages + [""] + spectrum_messages:
        print(line)

    overall = image_ok and spectrum_ok
    print("\nSUMMARY:", "PASS - ready to build." if overall else "FAIL - fix the items above before building.")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

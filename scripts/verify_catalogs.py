"""Verify the HATS catalogs named in a crossmatch config before running a real Phase 0 build.

Checks, all cheaply (catalog metadata only, no data download):

1. that the configured ``hats_path`` for each catalog opens;
2. that the configured top-level columns are present (image / spectrum structs, ra/dec/object_id,
   redshift, quality);
3. that the two catalogs actually overlap on the sky (the crossmatch alignment, which raises
   "Catalogs do not overlap" from metadata when they share no tiles).

It reads the same config keys that ``crossmatch/lsdb_match.py`` reads. Torch-free: needs ``lsdb``
(and pyyaml). Set a Hugging Face token first (``huggingface-cli login`` / ``HF_TOKEN``) to avoid 504s.
Exit code is non-zero if any required column is missing, a catalog fails to open, or they do not
overlap.

Usage:
    python scripts/verify_catalogs.py --config configs/crossmatch_legacy_desi.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _open_catalog(name: str, hats_path: str, columns: List[str], messages: List[str]):
    messages.append(f"[{name}] hats_path: {hats_path}")
    if hats_path.startswith("VERIFY"):
        messages.append(f"[{name}] FAIL: hats_path is still a placeholder - set the real hf:// path.")
        return None

    import lsdb

    try:
        return lsdb.open_catalog(hats_path, columns=columns)
    except Exception as exc:  # noqa: BLE001 - surface any open failure to the user
        messages.append(f"[{name}] FAIL: could not open catalog: {exc}")
        return None


def _check_columns(name: str, catalog, required: List[str], messages: List[str]) -> bool:
    available = list(catalog.columns)
    messages.append(f"[{name}] columns ({len(available)}): {available}")
    missing = [c for c in required if c not in available]
    if missing:
        messages.append(f"[{name}] FAIL: missing configured columns: {missing}")
        return False
    messages.append(f"[{name}] OK: all configured columns present.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify HATS catalogs before a Phase 0 build")
    parser.add_argument("--config", type=str, default="configs/crossmatch_legacy_desi.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    img = config["image_catalog"]
    spec = config["spectrum_catalog"]
    match = config.get("match", {})
    messages: List[str] = []

    image_required = [img["ra_column"], img["dec_column"], img["object_id_column"], img["image_column"]]
    spectrum_required = [
        spec["ra_column"], spec["dec_column"], spec["object_id_column"],
        spec["spectrum_column"], spec["redshift_column"],
    ]
    if spec.get("quality_column"):
        spectrum_required.append(spec["quality_column"])

    image_cat = _open_catalog("image", img["hats_path"], image_required, messages)
    spectrum_cat = _open_catalog("spectrum", spec["hats_path"], spectrum_required, messages)

    ok = image_cat is not None and spectrum_cat is not None
    if image_cat is not None:
        ok = _check_columns("image", image_cat, image_required, messages) and ok
    if spectrum_cat is not None:
        ok = _check_columns("spectrum", spectrum_cat, spectrum_required, messages) and ok

    if image_cat is not None and spectrum_cat is not None:
        try:
            image_cat.crossmatch(
                spectrum_cat,
                n_neighbors=int(match.get("n_neighbors", 1)),
                radius_arcsec=float(match.get("radius_arcsec", 1.0)),
                suffix_method="overlapping_columns",
            )
            messages.append("[overlap] OK: catalogs share sky tiles.")
        except Exception as exc:  # noqa: BLE001 - "Catalogs do not overlap" and any alignment error
            ok = False
            messages.append(f"[overlap] FAIL: {exc}")

    for line in messages:
        print(line)
    print("\nSUMMARY:", "PASS - ready to build." if ok else "FAIL - fix the items above before building.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

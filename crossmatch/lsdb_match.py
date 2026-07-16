"""Real cross-match path: open two MMU HATS catalogs and stream matched image+spectrum records.

Uses the documented LSDB API - ``lsdb.open_catalog(path, columns=...)`` then
``left.crossmatch(right, n_neighbors=1, radius_arcsec=...)`` - iterating the lazy result one
partition at a time so only overlapping sky tiles are materialized.

The MMU catalogs store image and spectrum data as nested structs, confirmed against the live
catalogs (``print(catalog)`` shows the schema without downloading):

- image catalog (e.g. ``UniverseTBD/mmu_ssl_legacysurvey_north``): column ``image`` is
  ``nested<band, flux, ...>`` where ``flux`` is the per-band (H, W) cutout, so ``image.flux``
  stacks to (n_bands, H, W).
- spectrum catalog (e.g. ``UniverseTBD/mmu_desi_edr_sv3``): column ``spectrum`` is
  ``nested<flux, ivar, lsf_sigma, lambda, mask>``, so ``spectrum.flux`` / ``spectrum.lambda`` are the
  1-D flux and wavelength; redshift is the top-level ``Z`` and the quality flag is ``ZWARN``.

Crossmatch keeps ``image`` / ``spectrum`` / ``Z`` / ``ZWARN`` unsuffixed; only ``ra`` / ``dec`` /
``object_id`` collide and get catalog-name suffixes, which are detected dynamically here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .schema import AlignedRecord

logger = logging.getLogger(__name__)


def _find_col(columns: List[str], base: str) -> Optional[str]:
    """Return ``base`` if present, else the first column named ``base_<suffix>`` (crossmatch suffix)."""
    if base in columns:
        return base
    for column in columns:
        if column.startswith(base + "_"):
            return column
    return None


def _nested_field(cell: Any, field: str) -> Any:
    """Access a sub-field of a nested cell, whether it materializes as a DataFrame or a mapping."""
    return cell[field]


def _stack_image(cell: Any, flux_field: str) -> np.ndarray:
    """Stack a nested image cell's per-band cutouts into a (n_bands, H, W) float32 array."""
    flux = _nested_field(cell, flux_field)
    bands = [np.asarray(band, dtype=np.float32) for band in flux]
    return np.stack(bands, axis=0)


def _to_1d(cell: Any, field: str) -> np.ndarray:
    """Extract a 1-D float32 array (flux or wavelength) from a nested spectrum cell."""
    value = _nested_field(cell, field)
    array = value.to_numpy() if hasattr(value, "to_numpy") else np.asarray(value)
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _is_bad_quality(value: Any) -> bool:
    """True when a quality flag marks a bad row (ZWARN != 0 / True)."""
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _iter_partitions(matched):
    """Yield computed partitions of a crossmatched catalog across LSDB versions."""
    frame = getattr(matched, "_ddf", matched)
    for delayed_partition in frame.to_delayed():
        yield delayed_partition.compute()


def crossmatched_records(config: Dict[str, Any], max_objects: Optional[int] = None) -> Iterator[AlignedRecord]:
    """Yield ``AlignedRecord`` objects from the cross-match described by ``config``."""
    import lsdb  # imported lazily so the synthetic smoke path needs no astro stack

    img_cfg = config["image_catalog"]
    spec_cfg = config["spectrum_catalog"]
    match_cfg = config["match"]

    image_columns = [
        img_cfg["ra_column"], img_cfg["dec_column"], img_cfg["object_id_column"], img_cfg["image_column"]
    ]
    spectrum_columns = [
        spec_cfg["ra_column"], spec_cfg["dec_column"], spec_cfg["object_id_column"],
        spec_cfg["spectrum_column"], spec_cfg["redshift_column"],
    ]
    quality_column = spec_cfg.get("quality_column")
    if quality_column:
        spectrum_columns.append(quality_column)

    logger.info("Opening image catalog: %s", img_cfg["hats_path"])
    image_cat = lsdb.open_catalog(img_cfg["hats_path"], columns=image_columns)
    logger.info("Opening spectrum catalog: %s", spec_cfg["hats_path"])
    spectrum_cat = lsdb.open_catalog(spec_cfg["hats_path"], columns=spectrum_columns)

    matched = image_cat.crossmatch(
        spectrum_cat,
        n_neighbors=int(match_cfg.get("n_neighbors", 1)),
        radius_arcsec=float(match_cfg.get("radius_arcsec", 1.0)),
        suffix_method="overlapping_columns",
    )

    image_col = img_cfg["image_column"]
    image_flux_field = img_cfg["image_flux_field"]
    spectrum_col = spec_cfg["spectrum_column"]
    flux_field = spec_cfg["flux_field"]
    wavelength_field = spec_cfg["wavelength_field"]
    redshift_col = spec_cfg["redshift_column"]

    n_yielded = 0
    n_skipped_quality = 0
    for partition in _iter_partitions(matched):
        if partition is None or len(partition) == 0:
            continue
        columns = list(partition.columns)
        oid_col = _find_col(columns, spec_cfg["object_id_column"]) or _find_col(columns, img_cfg["object_id_column"])
        ra_col = _find_col(columns, img_cfg["ra_column"])
        dec_col = _find_col(columns, img_cfg["dec_column"])

        for i in range(len(partition)):
            if quality_column and _is_bad_quality(partition[quality_column].iloc[i]):
                n_skipped_quality += 1
                continue
            image = _stack_image(partition[image_col].iloc[i], image_flux_field)
            spectrum_cell = partition[spectrum_col].iloc[i]
            flux = _to_1d(spectrum_cell, flux_field)
            wavelength = _to_1d(spectrum_cell, wavelength_field)
            redshift = float(partition[redshift_col].iloc[i])

            yield AlignedRecord(
                object_id=str(partition[oid_col].iloc[i]) if oid_col else f"{n_yielded:09d}",
                ra=float(partition[ra_col].iloc[i]) if ra_col else float("nan"),
                dec=float(partition[dec_col].iloc[i]) if dec_col else float("nan"),
                image=image,
                spectrum_flux=flux,
                spectrum_wavelength=wavelength,
                catalog={"redshift": redshift},
            )
            n_yielded += 1
            if max_objects is not None and n_yielded >= max_objects:
                logger.info("Reached max_objects=%d (skipped %d on quality).", max_objects, n_skipped_quality)
                return

    logger.info("Yielded %d matched records (skipped %d on quality).", n_yielded, n_skipped_quality)

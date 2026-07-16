# Phase 0 run guide: building the aligned dataset

Phase 0 cross-matches two Multimodal Universe HATS catalogs (Legacy Surveys images and DESI spectra)
on sky position and writes the aligned image-plus-spectrum dataset that Phases 1 and 2 consume.

## Where to run it

Phase 0 uses no GPU. There is no model in this path: `build_crossmatch.py` streams sky tiles through
LSDB and writes numpy shards. Memory is bounded by `output.shard_size` (the writer flushes each shard
to disk), so an 8 GB laptop is fine. The real constraints are disk space for the output and download
bandwidth.

Recommended split of work:

- Laptop (8 GB, no GPU): verify the catalogs and run one small trial build. Fast, free feedback.
- RunPod (where Phase 2 training runs): run the full build there, on the same instance you will train
  on, so the aligned dataset is co-located with training and you avoid uploading several GB. A CPU pod
  (or the CPU of your GPU pod) is enough; no GPU is needed for the build itself.

Rough sizing: each object stores a grz image cutout plus a DESI spectrum, on the order of 0.3 MB per
object (a rough estimate; measure it in the trial). So about 0.3 GB per 1,000 objects. A 20,000-object
build is a few GB. Pick `output.n_objects` accordingly.

## Step 1: install the Phase 0 dependencies (no torch)

Phase 0 does not import torch or transformers. On the machine doing the build:

```
pip install lsdb hats astropy huggingface_hub numpy pyyaml
```

## Step 2: verify the catalogs (do this first, on the laptop)

MMU HATS catalogs are opened from Hugging Face with the `hf://` protocol, for example
`lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_gz10")`. The config already points at:

- images: `hf://datasets/UniverseTBD/mmu_legacysurvey_test`
  (alternative: `hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north`)
- spectra: `hf://datasets/UniverseTBD/mmu_desi_edr_sv3`

To see the full list of available HATS repos (the source of truth for names):

```python
from huggingface_hub import HfApi
for d in HfApi().list_datasets(author="UniverseTBD"):
    if "mmu" in d.id:
        print(d.id)
```

Then run the verifier:

```
python scripts/verify_catalogs.py --config configs/crossmatch_legacy_desi.yaml
```

It opens both catalogs, prints their columns, checks that the configured column names
(image / flux / wavelength / redshift / ra / dec) exist, peeks one row to confirm the image and flux
cells are real arrays (not a coordinates-only catalog), and exits non-zero if anything is wrong. Fix
the column names in the config to match the printed columns until it prints `SUMMARY: PASS`. Before the
full build, sanity-check that the chosen image catalog actually overlaps DESI EDR by matching a small
sample and confirming a non-trivial match count.

If a catalog turns out to be coordinates-only, the arrays must be joined from the base
`MultimodalUniverse/legacysurvey` / `MultimodalUniverse/desi` datasets by id. That path is documented
in `crossmatch/lsdb_match.py` but not yet implemented; catching it here saves a wasted build.

## Step 3: trial build on the laptop

Set a small cap in `configs/crossmatch_legacy_desi.yaml`:

```yaml
output:
  n_objects: 1000     # trial cap
  shard_size: 256     # lower for a smaller RAM buffer
```

Run:

```
python build_crossmatch.py --config configs/crossmatch_legacy_desi.yaml
```

Then measure per-object size and inspect the output:

```
du -sh aligned/legacy_desi
head -1 aligned/legacy_desi/manifest.jsonl
python -c "import numpy as np; d=np.load('aligned/legacy_desi/shards/shard_00000.npz'); print({k: d[k].shape for k in d.files})"
```

Multiply the size for 1,000 objects by your target count to project the full dataset size.

## Step 4: full build on RunPod (co-located with training)

On the RunPod instance where Phase 2 will run:

```
git pull                                   # get your verified config
pip install lsdb hats astropy huggingface_hub numpy pyyaml
# raise output.n_objects to your target (for example 20000) in the config, then:
python build_crossmatch.py --config configs/crossmatch_legacy_desi.yaml
```

The aligned dataset lands under `aligned/legacy_desi/`, ready for `run_baseline.py`, `train.py`, and
`evaluate.py` on the same machine. From here, follow the real-run section of the top-level `README.md`.

## Tuning notes

- `output.n_objects` caps the build (keeps it laptop-scale for the trial, sized for the full run).
- `output.shard_size` controls the RAM buffer (objects held before a shard is flushed) and the number
  of shard files; lower it if memory is tight.
- `match.radius_arcsec` is 1.0 by default (standard optical same-object radius); `match.n_neighbors` is
  1 (nearest match only).
- `split.seed`, `split.val_fraction`, `split.test_fraction` control the deterministic, per-object
  train/val/test split written into the manifest.

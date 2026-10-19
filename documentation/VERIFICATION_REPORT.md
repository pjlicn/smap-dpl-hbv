# Package verification

Verification date: 2026-10-19.

## Checks performed

- All 23 Python source files compile successfully.
- Both training and testing entry points return successful `--help` output.
- Software metadata and example configuration parse successfully.
- The static-attribute matrix has shape 891 × 8, dtype `float64`, and the
  documented attribute order.
- Selected IDs, coordinates, and ordering agree across the packaged data files.
- The direct dPL-HBV architecture constructs successfully with 596,482
  parameters.
- The payload contains no checkpoints, Python caches, Git internals, or private
  absolute input paths.
- The manifest lists 46 payload files. SHA256 checksums cover 47 files,
  excluding the checksum file itself. The complete package contains 48 files.

Static-value SHA256 digest (C-order matrix bytes):

```text
979de5033f7449ad71d5440764ff18b109410c04b075883d64e6063424968278
```

## Run the checks

From the package root:

```bash
python -B scripts/verify_package.py
python -B code/train/train_smap_dpl_hbv.py --help
python -B code/test/test_smap_dpl_hbv.py --help
```

The package checks static-data integrity, metadata, imports, configuration,
and model construction. Numerical training, inference, gradient agreement,
and paper-result reproduction are outside this verification scope.

## Release consistency

This documentation update preserves all Python implementation files,
configuration, numerical static data, spatial indices, parameter settings,
software DOI, author order, and software version. ZIP and tar.gz contents are
verified against the package file inventory during release preparation.

Earlier verification records are retained in the repository history,
including commit `22699da0c1208774f8f211b0d19f8ead891cdb65`.

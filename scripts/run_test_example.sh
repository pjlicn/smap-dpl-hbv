#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${PACKAGE_ROOT}/code/test/test_smap_dpl_hbv.py" \
  --config "${PACKAGE_ROOT}/code/configs/example_config.yaml" \
  --dataset-npz /path/to/SMAP/jn04_dataset_select.npz \
  --pet-csv /path/to/SMAP/jn04_PEVAP.csv \
  --model-dir /path/to/model_directory \
  --output-dir /path/to/output

#!/usr/bin/env python3
"""Lightweight, self-contained verification for the release archive."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

sys.dont_write_bytecode = True


EXPECTED_NAMES = [
    "Albedo",
    "Capa",
    "landcover",
    "NDVI",
    "T_BULK_DEN",
    "T_CLAY",
    "T_SAND",
    "T_SILT",
]
EXPECTED_VALUE_DIGEST = "979de5033f7449ad71d5440764ff18b109410c04b075883d64e6063424968278"


def digest_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "zenodo.json",
        "environment.yml",
        "requirements.txt",
        "code/train/train_smap_dpl_hbv.py",
        "code/test/test_smap_dpl_hbv.py",
        "code/model/hydroDL/model/rnn.py",
        "code/model/hydroDL/model/HydroModels/HBV_Smap.py",
        "data/static_attributes/processed_static_attributes.nc",
        "data/spatial_indices/selected_locations.csv",
        "data/spatial_indices/selected_id_int64.npy",
    ]
    for relative in required:
        check((root / relative).is_file(), f"required file exists: {relative}")

    with (root / "zenodo.json").open(encoding="utf-8") as stream:
        zenodo = json.load(stream)
    check(len(zenodo["creators"]) == 6, "Zenodo metadata contains six ordered creators")
    check(zenodo["creators"][0]["name"] == "Li, Peijun", "first creator is Peijun Li")
    check(zenodo["creators"][-1]["name"] == "Lawson, Kathryn", "last creator is Kathryn Lawson")

    with (root / "CITATION.cff").open(encoding="utf-8") as stream:
        citation = yaml.safe_load(stream)
    check(citation["cff-version"] == "1.2.0", "CITATION.cff parses as CFF 1.2.0")
    check(len(citation["authors"]) == 6, "CITATION.cff contains six ordered authors")

    with (root / "code/configs/example_config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    check(set(config) == {"train", "test", "model"}, "example configuration has train/test/model sections")

    netcdf_path = root / "data/static_attributes/processed_static_attributes.nc"
    with xr.open_dataset(netcdf_path, engine="scipy") as dataset:
        values = dataset["static_attribute"].values
        names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in dataset["attribute_name"].values
        ]
        netcdf_ids = dataset["selected_id"].values.astype(np.int64)
        check(values.shape == (891, 8), "static matrix shape is 891 x 8")
        check(values.dtype == np.float64, "static matrix dtype is float64")
        check(names == EXPECTED_NAMES, "static attribute order matches model input")
        check(digest_array(values) == EXPECTED_VALUE_DIGEST, "static matrix digest matches extraction record")
        check(np.isfinite(dataset["longitude"].values).all(), "all selected longitudes are finite")
        check(np.isfinite(dataset["latitude"].values).all(), "all selected latitudes are finite")

    ids = np.load(root / "data/spatial_indices/selected_id_int64.npy", allow_pickle=False)
    locations = pd.read_csv(root / "data/spatial_indices/selected_locations.csv")
    check(ids.dtype == np.int64 and ids.shape == (891,), "canonical selected IDs retain int64 dtype")
    check(np.array_equal(ids, netcdf_ids), "NetCDF and NPY selected IDs agree")
    check(np.array_equal(ids, locations["selected_id"].to_numpy(dtype=np.int64)), "CSV and NPY selected IDs agree")
    check(np.unique(ids).size == 891, "selected IDs are unique")

    sys.path.insert(0, str(root / "code/model"))
    from hydroDL.model import rnn  # pylint: disable=import-outside-toplevel

    model = rnn.MultiInv_HBVTDModel_u30(
        ninv=14,
        nfea=16,
        nmul=16,
        hiddeninv=256,
        inittime=365,
        routOpt=False,
        comprout=False,
        compwts=False,
        staind=-1,
        tdlst=[1, 3, 13],
        dydrop=0.0,
        ETMod=True,
        model_name="HBV_Smap",
        model_name_sub="HBVMulTDET_1_1p_no_g_SMAP_ab",
    )
    check(model.ninv == 14 and model.nfea == 16 and model.nmul == 16, "primary architecture constructs")
    check(sum(parameter.numel() for parameter in model.parameters()) == 596482, "primary architecture parameter count is stable")

    forbidden = (
        "/data/" + "pql5336",
        "/home/" + "pql5336",
        "/data/" + "shared_data",
    )
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix in {".nc", ".npy"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(token in text for token in forbidden):
            offenders.append(str(path.relative_to(root)))
    check(not offenders, f"no private absolute paths found ({', '.join(offenders)})")

    forbidden_artifacts = [
        path for path in root.rglob("*")
        if path.is_file() and (path.suffix == ".pt" or "__pycache__" in path.parts or ".git" in path.parts)
    ]
    check(not forbidden_artifacts, "no checkpoints, Python caches, or Git internals are present")
    print("Verification completed successfully.")


if __name__ == "__main__":
    main()

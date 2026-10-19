#!/usr/bin/env python3
"""Extract the selected static attributes without changing values or order."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


EXPECTED_NAMES = np.array(
    ["Albedo", "Capa", "landcover", "NDVI", "T_BULK_DEN", "T_CLAY", "T_SAND", "T_SILT"]
)


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", required=True, help="Original jn04_dataset_select.npz.")
    parser.add_argument("--coordinates-csv", required=True, help="Source-grid crd.csv.")
    parser.add_argument("--output-netcdf", required=True, help="Destination NetCDF path.")
    parser.add_argument("--output-locations-csv", required=True, help="Destination selected-location CSV.")
    parser.add_argument("--output-selected-id-npy", required=True, help="Destination int64 selected-ID array.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = np.load(Path(args.source_npz), allow_pickle=False)
    required = {"data_st", "name_st", "selected_id"}
    missing = required.difference(source.files)
    if missing:
        raise ValueError(f"source NPZ is missing keys: {sorted(missing)}")

    names = source["name_st"].astype(str)
    selected_id = source["selected_id"].astype(np.int64, copy=False)
    values = source["data_st"].transpose(1, 0, 2).squeeze(axis=2)

    if values.shape != (891, 8):
        raise ValueError(f"expected static matrix (891, 8), found {values.shape}")
    if not np.array_equal(names, EXPECTED_NAMES):
        raise ValueError(f"unexpected attribute order: {names.tolist()}")
    if selected_id.shape != (891,) or np.unique(selected_id).size != 891:
        raise ValueError("selected_id must contain 891 unique entries")
    if values.dtype != np.float64:
        raise TypeError(f"expected float64 static values, found {values.dtype}")

    coordinates = pd.read_csv(Path(args.coordinates_csv))
    needed_columns = {"id", "lon", "lat"}
    if not needed_columns.issubset(coordinates.columns):
        raise ValueError(f"coordinate table requires columns {sorted(needed_columns)}")
    if selected_id.min() < 0 or selected_id.max() >= len(coordinates):
        raise IndexError("selected_id is outside the coordinate-table row range")
    selected = coordinates.iloc[selected_id].reset_index(drop=True)
    grid_id = selected["id"].to_numpy(dtype=np.int64)
    if not np.array_equal(grid_id, selected_id):
        raise ValueError("coordinate row index and grid id do not match")

    dataset = xr.Dataset(
        data_vars={
            "static_attribute": (("location", "attribute"), values),
            "selected_id": ("location", selected_id),
            "grid_id": ("location", grid_id),
            "longitude": ("location", selected["lon"].to_numpy(dtype=np.float64)),
            "latitude": ("location", selected["lat"].to_numpy(dtype=np.float64)),
            "attribute_name": ("attribute", names.astype("S10")),
        },
        coords={
            "location": np.arange(values.shape[0], dtype=np.int32),
            "attribute": np.arange(values.shape[1], dtype=np.int32),
        },
        attrs={
            "title": "Processed static geographic attributes for the SMAP dPL-HBV reference implementation",
            "history": "Extracted losslessly from jn04_dataset_select.npz",
            "row_order": "Original selected_id order; no sorting applied",
            "column_order": ", ".join(names.tolist()),
            "static_values_sha256": sha256_array(values),
            "selected_id_sha256": sha256_array(selected_id),
            "source_grid_resolution_degrees": 0.09336099584999147,
            "source_grid_is_rectangle": "false",
        },
    )
    dataset["static_attribute"].attrs.update(
        long_name="original processed static attribute value",
        note="Values and source encodings are preserved; no normalization is stored in this archive.",
    )
    dataset["longitude"].attrs.update(standard_name="longitude", units="degrees_east")
    dataset["latitude"].attrs.update(standard_name="latitude", units="degrees_north")

    output_netcdf = Path(args.output_netcdf)
    output_netcdf.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_netcdf, engine="scipy", format="NETCDF3_64BIT")

    output_csv = Path(args.output_locations_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "location": np.arange(values.shape[0], dtype=np.int32),
            "selected_id": selected_id,
            "grid_id": grid_id,
            "longitude": selected["lon"].to_numpy(dtype=np.float64),
            "latitude": selected["lat"].to_numpy(dtype=np.float64),
        }
    ).to_csv(output_csv, index=False, float_format="%.15g")

    output_id = Path(args.output_selected_id_npy)
    output_id.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_id, selected_id, allow_pickle=False)

    print(f"wrote {output_netcdf}")
    print(f"wrote {output_csv}")
    print(f"wrote {output_id}")
    print(f"static_values_sha256={sha256_array(values)}")


if __name__ == "__main__":
    main()

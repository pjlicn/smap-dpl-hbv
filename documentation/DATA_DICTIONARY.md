# Data dictionary

## Processed static attribute file

File: `data/static_attributes/processed_static_attributes.nc`

Dimensions:

- `location = 891`, in original selected-location order;
- `attribute = 8`, in model tensor order.

Variables:

| Variable | Type | Dimensions | Description |
|---|---|---|---|
| `static_attribute` | float64 | location × attribute | Original processed static values |
| `attribute_name` | fixed-length string | attribute | Attribute names in tensor order |
| `selected_id` | integer | location | Index into the source-grid tables |
| `grid_id` | integer | location | Verified source coordinate-table ID |
| `longitude` | float64 | location | Degrees east |
| `latitude` | float64 | location | Degrees north |

NetCDF-3 stores the included IDs as 32-bit integers; all IDs are below 100,000
and are represented exactly. The source `int64` array is separately preserved
as `data/spatial_indices/selected_id_int64.npy`.

The SHA-256 digest of the C-order bytes of the 891×8 `float64` matrix is:

```text
979de5033f7449ad71d5440764ff18b109410c04b075883d64e6063424968278
```

## Attribute fields

| Order | Name | Description | Units/source status | Observed range |
|---:|---|---|---|---:|
| 1 | Albedo | Surface albedo | Units and upstream product not documented | -9999 to 0.49 |
| 2 | Capa | Capa (source field label) | Units and upstream product not documented | -9999 to 1.8165 |
| 3 | landcover | Land-cover class | ESA CCI-LC / LCCS; 300 m native resolution (ESA, 2017); source year/version unconfirmed | 10 to 210 |
| 4 | NDVI | Normalized Difference Vegetation Index | Source convention/product not documented | -2166.380641666666 to 0.8381216666666665 |
| 5 | T_BULK_DEN | Soil bulk density | Units and upstream product not documented | 0 to 1.66 |
| 6 | T_CLAY | Soil clay fraction | Fraction convention/product not documented | 0 to 0.56 |
| 7 | T_SAND | Soil sand fraction | Fraction convention/product not documented | 0 to 0.86 |
| 8 | T_SILT | Soil silt fraction | Fraction convention/product not documented | 0 to 0.54 |

The values were selected from eight source-grid constant CSV files using
`selected_id`; no sorting, interpolation, normalization, or type conversion was
applied to the static matrix in this release. Source numeric encodings remain
present. Users adopting the reference architecture should define and document
their own quality-control and preprocessing policy.

The model input order is exactly the table order. No normalization means or
standard deviations are distributed.

## Spatial metadata

The selected points span approximately:

- longitude: -124.496887966805 to -67.4533195020747 degrees;
- latitude: 25.4907017177913 to 49.2724036712014 degrees.

The source coordinate metadata reports a nominal resolution of
0.09336099584999147 degrees and `isRectangle=false`. Row/column indices were
therefore not inferred. `selected_id`, verified `grid_id`, longitude, and
latitude provide the spatial linkage.

## Dynamic data schema (not included)

The source NPZ dynamic order is:

```text
APCP, DLWRF, DSWRF, PRES, SPFH, TMP, SMAP_AM, SMAP_SWI
```

Its shape is `8 × 891 × 1827` and dtype is `float64`. These dynamic data are
excluded from the archive.


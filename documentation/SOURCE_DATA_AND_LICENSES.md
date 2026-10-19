# Source data and licenses

## NLDAS-2 forcing

The atmospheric inputs use NLDAS-2 Primary Forcing Data L4 Hourly,
0.125° × 0.125°, product `NLDAS_FORA0125_H`: APCP, DLWRF, DSWRF, PRES, SPFH,
TMP, and PEVAP.

- Historical Version 002 identifier:
  <https://doi.org/10.5067/6J5LHHOHZHN4>
- Version 2.0 landing page:
  <https://disc.gsfc.nasa.gov/datasets/NLDAS_FORA0125_H_2.0/summary>
- Version 2.0 identifier:
  <https://doi.org/10.5067/THUF4J1RLSYG>

The PET-processing inputs use filenames matching
`NLDAS_FORA0125_H.*.002.grb.SUB.tif`, identifying the historical Version 002
lineage. Obtain the appropriate product from NASA Earthdata, subset the study
period and domain, and prepare daily inputs with the units expected by the
entry points. The forcing and PET time series are supplied separately from
this package.

## SMAP soil moisture

The study target is **SMAP Enhanced L3 Radiometer Global Daily 9 km EASE-Grid
Soil Moisture, Version 4**, using the **AM (morning)** retrieval, as identified
in the paper and SI Table S1. The product short name is `SPL3SMP_E`.

- Product DOI: <https://doi.org/10.5067/NJ34TQ2LFE90>
- Version 4 documentation: <https://nsidc.org/data/spl3smp_e/versions/4>
- Reference: O'Neill, P. E., Chan, S., Njoku, E. G., Jackson, T., Bindlish, R.,
  and Chaubell, J. (2020). *SMAP Enhanced L3 Radiometer Global Daily 9 km
  EASE-Grid Soil Moisture, Version 4*. NASA National Snow and Ice Data Center
  Distributed Active Archive Center.

The primary target field is `SMAP_AM`. The processed NPZ schema also contains
`SMAP_SWI`, a derived field whose construction is not specified by the included
entry points. The data schema covers 891 selected locations with daily labels
from 2015-04-01 through 2020-03-31. Version identification follows the paper's
data citation; original HDF5 version headers and the complete retrieval-quality
filtering history are not recorded in the packaged inputs.

SMAP observations are supplied separately. Use the cited product version when
preparing inputs for the study period; consult the provider for historical
product access.

## Processed static attributes

The package contains the selected values from eight source-grid constant
tables, with location IDs and coordinates. Field definitions and available
source metadata are provided in `DATA_DICTIONARY.md` and
`data/metadata/static_attribute_metadata.csv`. The full 96,591-location source
tables are separate from the selected 891-location release.

### Land cover

Land-cover classes use the **ESA Climate Change Initiative Land Cover
(CCI-LC)** classification based on the UN Land Cover Classification System
(LCCS). The product has a **nominal native spatial resolution of 300 m**.

Reference: ESA. (2017). *Land Cover CCI Product User Guide* (Version 2.0).
<https://maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf>

The reference year is the guide publication year. The source map year,
processing version, and original raster-to-grid procedure are not specified
in the retained input metadata. Native product resolution is distinct from
the study-grid spacing.

### Attribute metadata and use

Original processed values and numeric source encodings are retained.
For some attributes, upstream product identifiers, units, processing details,
and redistribution terms are not fully recorded in the available source
metadata. Field-level records distinguish documented values from unspecified
information. The code license does not establish permissions for upstream
third-party data products.

## Code licensing

Release-authored entry points, documentation, metadata, and utilities are
provided under BSD-3-Clause. Model components follow the project's hydroDL/δHBV
code lineage and retain their attribution. The available source records do
not establish a single upstream license covering every inherited model file;
this qualification remains separate from the license of release-authored
material. See the root `LICENSE` for the BSD-3-Clause terms.

# Source data: Urbanization erodes resilience in global mangroves

This dataset supports verification and reuse of the observations, scenario
comparisons and analytical calculations reported in the manuscript. It contains
the numerical sources for main Figures 2-6, Supplementary Figures 1-6 and
Supplementary Tables 1-12. Figure 1 is a conceptual diagram.

The data include individual neighbourhood observations, annual historical
estimates, all 15 paired projection members, component contributions, construction
accounts, spatial populations, validation results and analytical fields. True
zeros, positive and negative outcomes, missing observations and unknown forecast
coverage retain their distinct meanings.

Use `FIGURE_TABLE_INDEX.csv` to find a display, `FIELD_DEFINITIONS.txt` for units
and observation populations, and `FIELD_SCHEMA.csv` for individual file schemas.
Each file is included in `MANIFEST.json` with its SHA-256 digest.

| Directory | Content |
| --- | --- |
| main_figures | Annual trajectories, geographical context, paired effects and construction accounts |
| figure_story | Habitat and canopy observations, local construction example, component comparisons and analytical fields |
| reporting | Annual member states, paired differences, component contributions and cumulative effects |
| endpoint_2100 | Member and pixelwise endpoint summaries, affected areas, habitat changes and avoided-loss fractions |
| supplementary | Observation populations, evaluation results, annual regional estimates and sensitivity results |
| methods | Scientific definitions, calibration, coefficients and display parameters |
| urban_coverage | Selected city buffers, reference and global grid coordinates, spatial protocol and counts |
| city_buffer_sensitivity | Complete 1-, 2- and 5-km city-domain comparisons |

`figure_story/individual_neighbourhoods_2025.parquet` contains 925,801 individual
observations used for the historical distributions and retention-capacity
comparisons. Member tables retain member identity across years and development
choices. Quantiles and cumulative effects are calculated within the paired
design described in the manuscript.

For reproduction, place this directory under the name `SourceData` beside the
accompanying `SupplementarySoftware` directory, then follow its `README.txt`.
The software and fitted model demonstrations are available in the
SupplementarySoftware directory of this repository, version v1.2.4. Satellite and forcing archives are obtained from the original
providers listed in the manuscript; the spatial coverage protocol gives the
required product and tile identities.

Repository: https://github.com/FAITH-CKJ/urban-mangrove-resilience-data

Author-generated data are shared under Creative Commons Attribution 4.0
International. Cite this dataset and the associated manuscript when reusing
the data. Original third-party products retain their providers' licenses.
See `LICENSE.txt`, `CITATION.cff` and `THIRD_PARTY_SOURCES.txt`.

The journal Source Data attachment supplies a compact selection of these tables.
Its figure-table index links to this complete dataset for additional fields and spatial inputs.

The overlap of negative projected states and additional urban loss in 2100 is
supplied in `endpoint_2100/state_and_urban_loss_overlap_2100.csv`. Definitions
and denominators are in `endpoint_overlap_fields.txt` in the same directory.
Individual reference-cell states, paired effects and votes are available in the
complete repository as `endpoint_2100/pixelwise_median_endpoints_2100.npz`.

Supplementary Software: installation and reproduction

The package supplies figure/reporting code, the ecological forecasting model,
three fitted ecological checkpoints, the archived coastal classifier and its
seven checkpoints, labelled classifier training/test tables, and a real-data
forecast demonstration with expected production outputs.

1. Download version v1.2.4 of the public repository at
   https://github.com/FAITH-CKJ/urban-mangrove-resilience-data
   Keep its SourceData/ and SupplementarySoftware/ directories side by side.
   Figure reproduction uses this complete SourceData directory.
2. Use Python 3.11.7 and a virtual environment:
     python -m venv .venv
   Activate it using the command appropriate for the operating system, then:
     python -m pip install -r requirements.txt -r requirements-model.txt
3. From SupplementarySoftware, run:
     python code/verify_reporting.py
     python code/verify_figure_story.py
     python code/reproduce_forecast_demo.py
     python code/reproduce_main_figures.py
     python code/reproduce_supplementary.py

The forecast demonstration uses 16 real reference centres, the 2021-2025 encoder
and 75 annual predictions for 2026-2100. It replays fixed-city, urban-expansion
and habitat-protection branches for one shared SSP2-4.5 member. Expected outputs
come from the original production archives. The script checks all states, masks,
scores, five component values, contributions and quality flags, then writes NPZ
outputs and checks.json into reproduced/forecast_demo/. PASS and a maximum
absolute difference below 1e-7 are expected; archived outputs use float32.

Tested versions are recorded in tested_environment.json. These commands were tested
under Windows, Python 3.11.7, on an AMD Ryzen Threadripper PRO 5995WX workstation
with 128 GB RAM. The small forecast demonstration uses two CPU threads and
requires no GPU or other non-standard hardware. A 4-GB-memory desktop is
sufficient for this small demonstration. Package installation is estimated to
take 10-20 minutes on a desktop with a broadband connection; downloading PyTorch
dominates this estimate. The actual measured demo time is reported in
DEMO_RUNTIME.txt. Scientific figure regeneration can take several minutes.

Using other data
ecological_model/code/model_predict.py defines StateForecaster. Provide arrays
with the same names, feature order, units and shapes as ecological_model/demo/
metadata.json: five years of states and masks, five years of observed drivers,
future annual drivers, static attributes, and fixed H/K_B habitat counts.
Call predict_with_quality(..., capacity_support=H_K_array). The forecaster keeps
unknown inputs masked and projects ecological states onto their physical support.
Supplementary Method S6 defines all variables and transformations.

ecological_model/code/train_models.py records the training procedures for the
candidate models. The package supplies three fitted
checkpoints. code/reproduce_forecast_demo.py replays the central fit on the
supplied examples. Refitting requires the
historical centre-year table, driver archives, fixed geometry and complete
training protocol described in Supplementary Methods 6. MODEL_CARD.txt
documents the training, selection and evaluation populations and intended use.

The coastal classifier's training, inference and evaluation code is in
landcover_model/code/. For example:
  python landcover_model/code/evaluate_checkpoints.py --validation-csv
    landcover_model/training_data/test.csv.gz --models-dir
    landcover_model/models --output-json reproduced/landcover_metrics.json
Write this command on one line. Training has an explicit output-directory
argument so the distributed checkpoints can be retained.

Data access and licences

The Source Data dataset is maintained at
https://github.com/FAITH-CKJ/urban-mangrove-resilience-data
in the SourceData directory of version v1.2.4. The complete numerical dataset
is openly available without registration. The journal Source Data attachment
contains a compact selection, with links to the complete repository files. Author-generated data use CC BY 4.0 to enable
verification, reuse and further research. FIELD_DEFINITIONS.txt and FIGURE_TABLE_INDEX.csv describe the
data fields, populations and display correspondence.

The landcover_model code uses MIT; its sample tables and trained parameters use
CC BY 4.0 with the supplied attribution. Natural Earth cartography is public
domain. Other author code and ecological checkpoints are publicly available for
reproducing the reported analyses. Third-party dependencies retain their own licences.

Figure and table reproduction

Keep SourceData/ beside SupplementarySoftware/. Use Python 3.11 with requirements.txt.
Arial was used for the delivered figures. No network access or model fitting is
needed to reproduce the figures and tables from these source data.

From SupplementarySoftware run:
  python code/verify_reporting.py
  python code/verify_figure_story.py
  python code/reproduce_main_figures.py
  python code/reproduce_supplementary.py

Outputs are written only beneath SupplementarySoftware/reproduced/. The first
command independently reconstructs historical spatial summaries, member-paired
annual and cumulative quantiles, UMRI contribution budgets and allocation balances.
The main-figure command reproduces Figs. 2–6. The supplementary command
reproduces all six supplementary figures: observation and city-domain
sensitivity; regional scenario trajectories; historical vegetation and habitat
states; compensation boundaries with reconnection gains; and regional
protection outcomes with equal-construction mechanisms; and unsmoothed annual
histories on a common full scale. The twelve tables
retain classification, model validation, population and endpoint accounting.
Original annual, cumulative and component data remain in Source Data.
The supplementary command also writes its LaTeX source, bibliography and style.
Compile that source with XeLaTeX and BibTeX if a regenerated PDF is desired.

umri_core.py implements the five-component calculation and normalization defined in the manuscript.
umri_reporting.py derives final UMRI score columns and exact component budgets.
S and the weighted component fields are computational intermediates. The
calibration lambda and weights are fixed; the distributed predictions and
equation fields use annual recursion with the habitat-capacity constraints defined in Supplementary Methods 6. Spatial aggregation precedes the UMRI mapping; paired-member
decomposition precedes ensemble averaging. To regenerate final-score tables from the
equation fields, run python code/umri_reporting.py --source-data ../SourceData.
Regeneration writes derived columns and refreshes the derivation receipt.
The source integrity manifest refers to the distributed files; derive into a copy
when comparing a newly generated receipt with the distributed archive.
verify_reporting.py uses the supplied 577-centre sample to check component closure
and the exact gross-contribution calibration. Its primary reporting function averages the weighted intermediate score across
valid equal-area reference centres before applying the fixed UMRI normalization.

The full historical reconstruction and future state production use the source
archives listed in Methods. The distributed reporting tools reconstruct the
published figures and summaries from the supplied observations and derived tables.
Natural Earth map data are provided in cartography with source attribution.

Individual observations and analytical calculations

SourceData/figure_story/individual_neighbourhoods_2025.parquet contains the
individual habitat, recovery and canopy observations used in Figure 3 and the
retention-capacity comparisons. Recalculate the latter with:
  python code/derive_retention_reconnection_limit.py --output reproduced/retention_reconnection_limit.csv

Figure parameters and units are documented in SourceData/methods/figure3_parameters.json,
figure4_parameters.json, figure5_parameters.json and figure6_contours.json.
Use --figures 2,3,4,5,6 with reproduce_main_figures.py to choose figures, and
--draft to omit separate high-resolution raster exports. The default produces
the publication outputs; --output selects a separate output directory.

The paired construction accounts and relative habitat observations retain all
members, actual zero targets and undefined percentages. Figure 6 arrays include
the analytical axes, necessary habitat-support mask and balance curves.

Global historical habitat coverage
The compact coverage verification needs only numpy and pandas, and no downloads:
  python code/verify_urban_mangrove_coverage.py
It checks unique global/study coordinates, exact subset membership, all block totals,
regional counts, areas and the strict more-than-three-quarters criterion.

To recalculate coverage from native GMW rasters, install the additional geospatial
dependencies in requirements-coverage.txt and obtain gmw_mng_ext_v4112_stack.tar.gz
from https://doi.org/10.5281/zenodo.21346457. Place it in an inputs/ directory here,
or replace that argument below with its local location. The source archive's MD5,
size and required native-tile SHA-256 checksums are in the supplied protocol.
Run this command on one line:
  python code/compute_urban_mangrove_coverage.py --protocol ../SourceData/urban_coverage/coverage_protocol.json --buffers ../SourceData/urban_coverage/urban_buffers.gpkg --reference-cells ../SourceData/urban_coverage/reference_cells.npz --gmw-archive inputs/gmw_mng_ext_v4112_stack.tar.gz --output reproduced/urban_coverage --workers 4
The script verifies/extracts only the required 179 tiles, constructs the 2000-2004
presence union, and calculates 671 unique global cores. It supports verified cache
reuse within the specified output. Expected counts are 1204702 global and 934814
study cells, with zero unmatched study cells and 77.59711530320361% coverage.
Run fingerprints include protocol and code hashes, so paths or public provenance
metadata can change a fingerprint without changing the expected scientific results.

Optional: reconstruct the supplied city-buffer geometries from the original GUB2018
shapefile and its companion files, available from https://data-starcloud.pcl.ac.cn/iearthdata/14:
  python code/prepare_urban_mangrove_buffers.py --gub inputs/GUB_Global_2018.shp --catalog ../SourceData/urban_coverage/gmw_v4112_tiles.geojson --output reproduced/urban_coverage/urban_buffers.gpkg
The geometry settings are 30 m topology-preserving simplification and 2000 m
buffer in EPSG:6933, with quad_segs=2. All original polygons with urbanArea >100
are considered before selecting the buffers intersecting the global tile catalogue.


Individual-observation and model-evaluation checks

  python code/verify_neighbourhood_observations.py
  python code/verify_table5.py

These checks reconstruct the 33 Figure 3 distributions, the 110 retention-limit
comparisons and all eight model/persistence holdout evaluation columns from the
supplied observations and error components.

Endpoint overlap in Supplementary Table 9

Run python code/derive_endpoint_overlap.py to reconstruct the intersection of
negative projected UMRI and additional urban loss at the same reference cells.
The calculation reads endpoint_2100/pixelwise_median_endpoints_2100.npz from the
complete SourceData directory and checks the supplied 88-row summary exactly.
The supplementary-table builder also performs this check before rendering Table 9.
Definitions and denominators are in endpoint_2100/endpoint_overlap_fields.txt.

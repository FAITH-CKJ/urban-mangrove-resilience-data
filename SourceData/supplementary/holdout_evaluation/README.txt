Supplementary Table 5: underlying holdout error records

These files supply the error totals, observation counts and weighting
components needed to reproduce every numerical entry in Supplementary
Table 5. The records derive from the five-year and ten-year holdout predictions
and their historical observations. The persistence comparator uses causal
carry-forward and the habitat constraints described in the manuscript.

Evaluation populations and periods

Both horizons are scored on 2021–2025. The five-year model starts in 2020.
The ten-year model starts in 2015, runs recursively through 2025, and is
scored only over its terminal five years. The temporal evaluation uses
55,273 uniformly sampled centres in the non-held-out blocks; the spatial
evaluation uses 2,625 uniformly sampled centres in ten held-out blocks.
Rare-state enrichment centres are not part of this evaluation population.
The source keys identify horizon, split, method, macroregion and spatial
block. A block ID is a scientific spatial key. These ecological
neighbourhoods can overlap and are not independent treatment replicates.

Files

holdout_score_pairs_by_block_year.csv
  One row per method, horizon, split, block and scored year. valid_centres
  is the number of jointly valid score records. predicted_S_sum and
  observed_S_sum are their computational-intermediate totals;
  absolute_S_error_sum retains the sum of their individual absolute errors.
  Dividing the first two totals by valid_centres gives their block means,
  also provided as predicted_S_mean and observed_S_mean.

holdout_state_error_components.csv
  One row per method, horizon, split, state, block and scored year.
  valid_centre_years counts finite state errors in that year.
  absolute_error_sum is the unweighted sum of their absolute errors.
  For each centre i, let n_i be its number of finite errors over 2021–2025.
  centre_year_normalized_absolute_error_sum stores the sum of error_i,t/n_i
  within the block at year t. eligible_centres counts the centres with
  n_i>0 for that state and block and is constant across its five rows.
  Sum the normalized totals over years and divide by eligible_centres to
  obtain the mean of centre-specific temporal MAEs within that block.
  This preserves the reported ordering even when centres have different
  numbers of valid years; pooling all raw errors and dividing by the total
  number of centre-years would be a different estimator.

holdout_direction_pairs_by_block.csv
  Counts and S totals/means for the same valid centres at the observed
  2020 origin and 2025 endpoint. The four quantities are observed_origin_S,
  predicted_origin_S, observed_terminal_S and predicted_terminal_S.
  In the five-year test, predicted_origin_S equals the observed origin.
  In the ten-year test, it is the recursively predicted state at 2020.
  Differences are classified using a zero tolerance of 1e-8 in S units.

holdout_absence_counts_by_block_year.csv
  Four presence/absence count categories. True and predicted absence are
  retained explicitly. These counts use the observed retention-state mask,
  rather than requiring all five components to yield a valid score.
  Their totals can therefore differ from valid_centres in the score table.
  False absence is true_present_pred_absent.

../forecast_holdout_metrics.csv
  The full-precision reported summaries for the eight combinations of
  horizon, temporal/spatial split and model/persistence. It is the numerical
  comparison target, not the basis from which the error records were made.

evaluation_definitions.json
  Normalization, direction tolerance, scoring years, state order,
  evaluation populations and exact aggregation definitions.

SupplementarySoftware/code/verify_table5.py
  Independently reconstructs all five state MAEs, S MAE, UMRI MAE,
  direction accuracy, false absence, valid score records and population
  counts from these files. It reads no external data and does not fit or
  run any model. Run with Python, NumPy and pandas:

      python code/verify_table5.py

  Run from the SupplementarySoftware directory placed beside SourceData.
  An optional --output path writes the detailed verification result.

Aggregation and units

For score errors, first obtain centre-weighted means within each block and
year; give blocks equal weight within a region and year. Calculate S error
from those regional means. Calculate UMRI error after applying
N(S)=S/(0.05934161890467785+abs(S)) separately to the predicted and observed
regional mean. Average the annual errors within regions, then give regions
equal weight. This differs from averaging individual normalized scores or
individual absolute errors.

For state errors, first average each centre's absolute errors over its own
valid years; give centres equal weight within blocks, blocks equal weight
within regions, and regions equal weight. The normalized-error components
preserve this ordering without publishing the model-training feature panel.

For direction, average the four values within each block on the common
origin–endpoint mask, give blocks equal weight within regions, then compare
the signs of predicted and observed regional changes. Direction accuracy is
the fraction of regions with matching signs. False absences are record
counts and are summed, not averaged.

m, C and recovered cover are fractions; EVI and NDMI are dimensionless
indices. S is the weighted computational intermediate. UMRI is the final
dimensionless normalized score. Error totals use the corresponding state
or score units. Counts are integers. Zero error and zero counts are real
zeros. NA in a mean or total denotes no valid supporting observations;
zero valid or eligible counts must not be converted to a zero error mean.

All decimal values are retained at round-trip double precision. The
verification checks integer counts exactly and floating-point summaries
within an absolute tolerance of 1e-12, allowing only arithmetic summation
and serialization differences far below the displayed precision.

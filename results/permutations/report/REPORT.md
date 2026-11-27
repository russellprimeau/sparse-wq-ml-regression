# Block-permutation experiment

Original and rearranged records repeat complete selection within each specified arm. Three consecutive laboratory observations move together across fixed dates; environmental records remain fixed. Laboratory priors, differential responses and six references are rebuilt.

10 completed numbered rearrangements per target; 10 specified independently of scores.

Each comparison uses matching primary evaluation dates. Improvement (%) is 100 × (1 − model RMSE / lowest RMSE among six references), computed separately for each record. These are descriptive comparisons conditional on the specified arms and exchangeable observation blocks.

## Total coliforms

noprofiler_differential

Original R² higher in 10/10 comparisons; original improvement against references higher in 10/10.

| Repetition | Shared / primary dates | Original R² | Rearranged R² | Original RMSE | Rearranged RMSE | Original improvement (%) | Rearranged improvement (%) | Selected model |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 50/50 | 0.5518 | 0.1861 | 6.443 | 6.159 | 5.378 | -8.306 | Multiple linear regression (latest input) |
| 2 | 50/50 | 0.5518 | -0.08281 | 6.443 | 8.441 | 5.378 | -3.748 | XGBoost (24-hour summaries) |
| 3 | 50/50 | 0.5518 | -0.1226 | 6.443 | 6.99 | 5.378 | -9.219 | XGBoost (12-hour summaries) |
| 4 | 50/50 | 0.5518 | -0.07956 | 6.443 | 8.643 | 5.378 | -9.992 | XGBoost (12-hour summaries) |
| 5 | 50/50 | 0.5518 | -0.1338 | 6.443 | 7.352 | 5.378 | -5.851 | Multiple linear regression (latest input) |
| 6 | 50/50 | 0.5518 | -0.3278 | 6.443 | 8.375 | 5.378 | -20.72 | Gaussian process (Matérn 5/2 + linear; full-window means) |
| 7 | 50/50 | 0.5518 | -0.6132 | 6.443 | 9.618 | 5.378 | -23.78 | XGBoost (12-hour summaries) |
| 8 | 50/50 | 0.5518 | -0.07461 | 6.443 | 6.634 | 5.378 | 2.761 | Gaussian process (Matérn 5/2; full-window means) |
| 9 | 50/50 | 0.5518 | 0.05183 | 6.443 | 7.375 | 5.378 | -9.247 | Gaussian process (Matérn 5/2 + linear; full-window means) |
| 10 | 50/50 | 0.5518 | 0.08006 | 6.443 | 6.991 | 5.378 | 1.391 | Multiple linear regression (latest input) |

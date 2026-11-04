# Confirmatory results

Pre-registered in `hypotheses.json`. Seed 31337, 200 truth models per scenario, run time 691 s. Differences are method minus semivariogram regret in $MM per 3-well program (negative = method better), averaged within each truth model. Holm correction over H1-H7; equivalence margin +/- 0.25 $MM (90% CI).

## Pre-registered hypotheses

| id | claim | n_truths | mean_diff_MM | ci95_lo | ci95_hi | p | p_holm | pct_truths_better | verdict |
|---|---|---|---|---|---|---|---|---|---|
| H1 | Picking the type by leave-one-out cross-validation lowers regret in the facies scenario | 200 | -0.130 | -0.270 | 0.010 | 0.069 | 0.411 | 48.000 | EQUIVALENT |
| H2 | The normal-score semivariogram lowers regret for skewed porosity | 200 | 0.049 | -0.088 | 0.186 | 0.478 | 0.664 | 46.500 | EQUIVALENT |
| H3 | Cross-validation pick lowers regret at 25 wells across the four main-study scenarios | 800 | -0.111 | -0.188 | -0.034 | 0.005 | 0.034 | 49.750 | CONFIRMED |
| H4 | The pairwise relative variogram lowers regret when 10% of well values are gross errors | 200 | -0.056 | -0.145 | 0.034 | 0.221 | 0.664 | 51.000 | EQUIVALENT |
| H5 | The rodogram lowers regret when 10% of well values are gross errors | 200 | -0.067 | -0.157 | 0.024 | 0.147 | 0.628 | 51.000 | EQUIVALENT |
| H6 | The general relative variogram lowers regret under a strong proportional effect | 200 | -0.116 | -0.264 | 0.033 | 0.126 | 0.628 | 49.500 | EQUIVALENT |
| H7 | The categorical (facies) indicator variogram, gamv type 10, lowers regret in the facies scenario | 200 | -0.067 | -0.196 | 0.063 | 0.311 | 0.664 | 46.000 | EQUIVALENT |

## Mean regret by scenario ($MM per program, descriptive)

| scenario | semivariogram | normal_score | pairwise_relative | rodogram | general_relative | loocv | indicator_continuous | indicator_categorical | truth_variogram | idw |
|---|---|---|---|---|---|---|---|---|---|---|
| anisotropic, 25 wells | 2.021 | 2.030 | 2.004 | 2.021 | 1.991 | 1.910 | 2.081 | nan | 1.574 | 1.831 |
| facies, 25 wells | 4.374 | 4.402 | 4.316 | 4.321 | 4.318 | 4.155 | 4.216 | 4.314 | 3.333 | 3.912 |
| facies, 50 wells | 2.737 | 2.717 | 2.662 | 2.630 | 2.764 | 2.607 | 2.660 | 2.670 | 2.310 | 2.474 |
| gaussian, 25 wells | 1.991 | 2.051 | 2.010 | 1.989 | 2.015 | 1.929 | 1.955 | nan | 1.485 | 1.772 |
| outliers, 50 wells | 1.915 | 1.817 | 1.859 | 1.848 | 1.858 | 1.782 | 1.789 | nan | 1.546 | 1.642 |
| proportional, 50 wells | 2.576 | 2.431 | 2.402 | 2.495 | 2.460 | 2.269 | 2.604 | nan | 2.058 | 2.079 |
| skewed, 25 wells | 4.041 | 3.941 | 3.941 | 3.933 | 4.010 | 3.988 | 4.111 | nan | 3.187 | 3.606 |
| skewed, 50 wells | 2.298 | 2.347 | 2.205 | 2.345 | 2.241 | 2.114 | 2.473 | nan | 1.739 | 2.092 |

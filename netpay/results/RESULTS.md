# Net-pay and maximum-likelihood results

Seed 60606, 100 truth models per scenario, run time 2057 s. Differences are A minus B (negative = A better), averaged within each truth model over both well designs, at [25, 50, 100] wells unless the hypothesis says otherwise. Holm correction over all hypotheses. Regret in $MM per 3-well program; net-OOIP error in % of the true field net OOIP.

## Pre-registered hypotheses

| id | claim | n_truths | mean_diff | ci95_lo | ci95_hi | p_holm | pct_truths_better | verdict |
|---|---|---|---|---|---|---|---|---|
| H1 | With a net-pay cutoff, deciding on the expected net OOIP beats deciding on the smooth kriged map | 400 | -0.515 | -0.595 | -0.435 | 0.000 | 73.250 | CONFIRMED |
| H2 | The expected net OOIP estimates field net OOIP better than the kriged map with the cutoff applied | 400 | -15.068 | -16.355 | -13.781 | 0.000 | 88.000 | CONFIRMED |
| H3 | Maximum-likelihood variogram beats the fitted semivariogram (original decision) | 400 | 0.031 | -0.036 | 0.097 | 1.000 | 52.750 | EQUIVALENT |
| H4 | Maximum-likelihood variogram beats the fitted semivariogram (net-pay decision) | 400 | -0.093 | -0.191 | 0.005 | 0.505 | 56.500 | EQUIVALENT |
| H5 | Constraining the fit (minimum range, anisotropy cap) beats the unconstrained fit | 400 | -0.032 | -0.055 | -0.010 | 0.046 | 52.750 | CONFIRMED |
| H6 | Kriging with the constrained fit beats inverse distance | 400 | 0.114 | 0.060 | 0.168 | 0.000 | 43.750 | REVERSED |
| H7 | Correlogram differs from the semivariogram under the net-pay decision | 400 | -0.050 | -0.130 | 0.031 | 1.000 | 51.000 | EQUIVALENT |
| H8 | General relative differs from the semivariogram under the net-pay decision | 400 | -0.021 | -0.078 | 0.037 | 1.000 | 54.500 | EQUIVALENT |
| H9 | Pairwise relative differs from the semivariogram under the net-pay decision | 400 | 0.010 | -0.054 | 0.074 | 1.000 | 52.000 | EQUIVALENT |
| H10 | Log semivariogram differs from the semivariogram under the net-pay decision | 400 | 0.028 | -0.036 | 0.091 | 1.000 | 50.000 | EQUIVALENT |
| H11 | Rodogram differs from the semivariogram under the net-pay decision | 400 | -0.065 | -0.134 | 0.004 | 0.505 | 57.250 | EQUIVALENT |
| H12 | Cumulative differs from the semivariogram under the net-pay decision | 400 | -0.124 | -0.196 | -0.053 | 0.007 | 58.500 | CONFIRMED |
| H13 | Normal-score semivariogram differs from the semivariogram under the net-pay decision | 400 | 0.012 | -0.059 | 0.083 | 1.000 | 50.750 | EQUIVALENT |

## Mean regret ($MM per program) by rule and well count

`map_total`: wells placed on the kriged map, all OOIP counts (the main study's decision). `map_net`: wells placed on the kriged map, net OOIP only. `etype_net`: wells placed on the expected net OOIP. `sgs_net`: the same from 50 SGS realizations.

| method | etype_net, 25 | etype_net, 50 | etype_net, 100 | map_net, 25 | map_net, 50 | map_net, 100 | map_total, 25 | map_total, 50 | map_total, 100 | sgs_net, 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| semivariogram | 5.216 | 3.424 | 1.999 | 5.890 | 3.927 | 2.368 | 3.022 | 1.902 | 1.105 | 3.443 |
| semivariogram_c | 5.097 | 3.327 | 1.974 | 5.731 | 3.817 | 2.319 | 2.955 | 1.888 | 1.090 | nan |
| ml | 5.459 | 3.260 | 1.643 | 6.003 | 3.605 | 1.909 | 3.244 | 1.897 | 0.981 | nan |
| correlogram | 5.222 | 3.332 | 1.937 | 5.864 | 3.740 | 2.305 | 3.031 | 1.875 | 1.115 | nan |
| general_relative | 5.182 | 3.407 | 1.988 | 5.959 | 3.764 | 2.256 | 3.038 | 1.900 | 1.099 | nan |
| pairwise_relative | 5.276 | 3.372 | 2.024 | 6.044 | 3.829 | 2.289 | 3.078 | 1.950 | 1.091 | nan |
| log | 5.308 | 3.409 | 2.006 | 6.036 | 3.843 | 2.249 | 3.085 | 1.944 | 1.097 | nan |
| rodogram | 5.135 | 3.296 | 2.014 | 5.959 | 3.655 | 2.285 | 3.028 | 1.893 | 1.113 | nan |
| cumulative | 5.244 | 3.262 | 1.761 | 6.060 | 3.716 | 2.014 | 3.072 | 1.817 | 0.985 | nan |
| normal_score | 5.245 | 3.358 | 2.073 | 6.008 | 3.854 | 2.241 | 3.034 | 1.952 | 1.117 | nan |
| truth_variogram | 4.265 | 2.778 | 1.612 | 4.930 | 3.118 | 1.876 | 2.326 | 1.533 | 0.911 | nan |
| idw | nan | nan | nan | 5.609 | 3.774 | 2.432 | 2.740 | 1.788 | 1.063 | nan |

## Field net OOIP: mean absolute error (%)

| method | etype_net, 25 | etype_net, 50 | etype_net, 100 | map_net, 25 | map_net, 50 | map_net, 100 | sgs_net, 50 |
|---|---|---|---|---|---|---|---|
| semivariogram | 35.991 | 28.181 | 21.790 | 63.335 | 40.549 | 27.282 | 28.298 |
| semivariogram_c | 36.067 | 27.972 | 21.659 | 49.848 | 36.251 | 26.532 | nan |
| ml | 36.581 | 27.304 | 20.273 | 50.551 | 34.216 | 22.893 | nan |
| correlogram | 36.152 | 28.142 | 22.182 | 69.522 | 43.966 | 28.876 | nan |
| general_relative | 35.568 | 27.752 | 20.891 | 61.616 | 40.628 | 27.071 | nan |
| pairwise_relative | 35.612 | 27.757 | 21.251 | 62.577 | 40.417 | 27.040 | nan |
| log | 35.583 | 27.698 | 21.000 | 61.485 | 39.647 | 26.639 | nan |
| rodogram | 36.032 | 28.045 | 21.694 | 66.282 | 39.119 | 24.419 | nan |
| cumulative | 36.129 | 28.026 | 21.394 | 66.546 | 40.975 | 25.664 | nan |
| normal_score | 36.256 | 28.349 | 21.709 | 62.841 | 39.858 | 27.950 | nan |
| truth_variogram | 35.153 | 26.353 | 19.590 | 51.077 | 33.830 | 23.568 | nan |
| idw | nan | nan | nan | 53.827 | 45.794 | 42.009 | nan |

## Variogram model accuracy: azimuth error (deg, anisotropic scenarios) and short-range RMSE vs truth

| method | azi_err, 25 | azi_err, 50 | azi_err, 100 | vario_rmse, 25 | vario_rmse, 50 | vario_rmse, 100 |
|---|---|---|---|---|---|---|
| semivariogram | 35.725 | 27.950 | 21.000 | 0.277 | 0.193 | 0.145 |
| semivariogram_c | 35.725 | 27.950 | 21.000 | 0.246 | 0.179 | 0.142 |
| ml | 32.998 | 20.526 | 11.366 | 0.225 | 0.167 | 0.130 |
| correlogram | 36.825 | 28.050 | 20.100 | 0.270 | 0.190 | 0.147 |
| general_relative | 36.525 | 27.375 | 20.875 | 0.276 | 0.192 | 0.145 |
| pairwise_relative | 36.000 | 27.075 | 20.375 | 0.275 | 0.190 | 0.143 |
| log | 36.400 | 27.100 | 20.675 | 0.278 | 0.193 | 0.145 |
| rodogram | 36.250 | 27.450 | 20.750 | 0.274 | 0.205 | 0.160 |
| cumulative | 35.300 | 27.075 | 17.975 | 0.291 | 0.199 | 0.147 |
| normal_score | 36.900 | 28.875 | 21.425 | 0.268 | 0.191 | 0.147 |
| truth_variogram | 9.700 | 9.700 | 9.700 | 0.077 | 0.077 | 0.077 |
| idw | nan | nan | nan | nan | nan | nan |

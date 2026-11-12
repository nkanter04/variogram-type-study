# Net-pay and maximum-likelihood results (EXPLORATORY PILOT - not a finding)

Seed 555, 10 truth models per scenario, run time 485 s. Differences are A minus B (negative = A better), averaged within each truth model over both well designs, at [25, 50, 100] wells unless the hypothesis says otherwise. Holm correction over all hypotheses. Regret in $MM per 3-well program; net-OOIP error in % of the true field net OOIP.

## Pre-registered hypotheses

| id | claim | n_truths | mean_diff | ci95_lo | ci95_hi | p_holm | pct_truths_better | verdict |
|---|---|---|---|---|---|---|---|---|
| H1 | With a net-pay cutoff, deciding on the expected net OOIP beats deciding on the smooth kriged map | 40 | -0.227 | -0.441 | -0.013 | 0.381 | 62.500 | INCONCLUSIVE |
| H2 | The expected net OOIP estimates field net OOIP better than the kriged map with the cutoff applied | 40 | -13.158 | -17.386 | -8.930 | 0.000 | 90.000 | CONFIRMED |
| H3 | Maximum-likelihood variogram beats the fitted semivariogram (original decision) | 40 | -0.090 | -0.296 | 0.116 | 1.000 | 55.000 | INCONCLUSIVE |
| H4 | Maximum-likelihood variogram beats the fitted semivariogram (net-pay decision) | 40 | -0.171 | -0.412 | 0.070 | 1.000 | 55.000 | INCONCLUSIVE |
| H5 | Constraining the fit (minimum range, anisotropy cap) beats the unconstrained fit | 40 | -0.105 | -0.192 | -0.018 | 0.225 | 55.000 | EQUIVALENT |
| H6 | Kriging with the constrained fit beats inverse distance | 40 | 0.157 | 0.026 | 0.289 | 0.225 | 37.500 | INCONCLUSIVE |
| H7 | Correlogram differs from the semivariogram under the net-pay decision | 40 | -0.053 | -0.352 | 0.246 | 1.000 | 57.500 | INCONCLUSIVE |
| H8 | General relative differs from the semivariogram under the net-pay decision | 40 | 0.086 | -0.063 | 0.235 | 1.000 | 30.000 | EQUIVALENT |
| H9 | Pairwise relative differs from the semivariogram under the net-pay decision | 40 | 0.030 | -0.137 | 0.198 | 1.000 | 37.500 | EQUIVALENT |
| H10 | Log semivariogram differs from the semivariogram under the net-pay decision | 40 | -0.016 | -0.183 | 0.150 | 1.000 | 50.000 | EQUIVALENT |
| H11 | Rodogram differs from the semivariogram under the net-pay decision | 40 | -0.174 | -0.356 | 0.009 | 0.557 | 60.000 | INCONCLUSIVE |
| H12 | Cumulative differs from the semivariogram under the net-pay decision | 40 | -0.158 | -0.385 | 0.070 | 1.000 | 65.000 | INCONCLUSIVE |
| H13 | Normal-score semivariogram differs from the semivariogram under the net-pay decision | 40 | -0.025 | -0.200 | 0.150 | 1.000 | 52.500 | EQUIVALENT |

## Mean regret ($MM per program) by rule and well count

`map_total`: wells placed on the kriged map, all OOIP counts (the main study's decision). `map_net`: wells placed on the kriged map, net OOIP only. `etype_net`: wells placed on the expected net OOIP. `sgs_net`: the same from 50 SGS realizations.

| method | etype_net | 25 | etype_net | 50 | etype_net | 100 | map_net | 25 | map_net | 50 | map_net | 100 | map_total | 25 | map_total | 50 | map_total | 100 | sgs_net | 25 | sgs_net | 50 | sgs_net | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| semivariogram | 5.672 | 3.372 | 2.237 | 5.847 | 3.688 | 2.426 | 3.181 | 1.854 | 1.198 | 5.533 | 3.498 | 2.235 |
| semivariogram_c | 5.195 | 3.259 | 2.188 | 5.537 | 3.521 | 2.342 | 3.042 | 1.699 | 1.176 | nan | nan | nan |
| ml | 5.497 | 3.175 | 2.096 | 5.922 | 3.678 | 2.167 | 3.293 | 1.609 | 1.060 | nan | nan | nan |
| correlogram | 5.671 | 3.250 | 2.200 | 6.019 | 4.031 | 2.308 | 3.280 | 1.746 | 1.230 | nan | nan | nan |
| general_relative | 5.565 | 3.707 | 2.265 | 5.773 | 3.648 | 2.381 | 3.153 | 2.088 | 1.233 | nan | nan | nan |
| pairwise_relative | 5.568 | 3.503 | 2.299 | 5.699 | 3.832 | 2.455 | 3.105 | 1.951 | 1.197 | nan | nan | nan |
| log | 5.457 | 3.476 | 2.298 | 5.669 | 3.873 | 2.479 | 3.080 | 1.979 | 1.191 | nan | nan | nan |
| rodogram | 5.160 | 3.349 | 2.249 | 5.369 | 3.656 | 2.270 | 2.929 | 2.029 | 1.187 | nan | nan | nan |
| cumulative | 5.291 | 3.330 | 2.187 | 5.615 | 3.639 | 2.395 | 2.966 | 1.697 | 1.149 | nan | nan | nan |
| normal_score | 5.655 | 3.223 | 2.327 | 5.773 | 3.653 | 2.470 | 3.017 | 1.833 | 1.255 | nan | nan | nan |
| truth_variogram | 4.132 | 2.986 | 1.785 | 4.537 | 3.330 | 1.946 | 2.352 | 1.570 | 0.939 | nan | nan | nan |
| idw | nan | nan | nan | 5.012 | 3.915 | 2.749 | 2.651 | 1.590 | 1.205 | nan | nan | nan |

## Field net OOIP: mean absolute error (%)

| method | etype_net | 25 | etype_net | 50 | etype_net | 100 | map_net | 25 | map_net | 50 | map_net | 100 | sgs_net | 25 | sgs_net | 50 | sgs_net | 100 |
|---|---|---|---|---|---|---|---|---|---|
| semivariogram | 34.221 | 28.504 | 20.612 | 59.746 | 38.043 | 25.022 | 34.173 | 28.573 | 20.585 |
| semivariogram_c | 34.186 | 28.620 | 20.503 | 47.569 | 34.626 | 24.308 | nan | nan | nan |
| ml | 33.664 | 28.920 | 20.152 | 49.609 | 29.482 | 20.755 | nan | nan | nan |
| correlogram | 33.567 | 28.207 | 21.215 | 63.597 | 42.670 | 27.448 | nan | nan | nan |
| general_relative | 33.765 | 27.677 | 19.794 | 56.705 | 39.501 | 23.999 | nan | nan | nan |
| pairwise_relative | 33.626 | 28.015 | 19.945 | 58.388 | 38.403 | 25.982 | nan | nan | nan |
| log | 33.726 | 27.811 | 19.814 | 56.822 | 38.217 | 25.665 | nan | nan | nan |
| rodogram | 33.246 | 28.068 | 20.598 | 60.183 | 38.032 | 21.717 | nan | nan | nan |
| cumulative | 33.862 | 28.907 | 21.010 | 62.054 | 41.374 | 23.806 | nan | nan | nan |
| normal_score | 33.219 | 28.149 | 20.830 | 60.924 | 39.977 | 26.872 | nan | nan | nan |
| truth_variogram | 32.177 | 27.337 | 19.192 | 48.880 | 30.453 | 21.800 | nan | nan | nan |
| idw | nan | nan | nan | 50.335 | 43.634 | 41.004 | nan | nan | nan |

## Variogram model accuracy: azimuth error (deg, anisotropic scenarios) and short-range RMSE vs truth

| method | azi_err | 25 | azi_err | 50 | azi_err | 100 | vario_rmse | 25 | vario_rmse | 50 | vario_rmse | 100 |
|---|---|---|---|---|---|---|
| semivariogram | 37.000 | 27.250 | 21.000 | 0.274 | 0.177 | 0.141 |
| semivariogram_c | 37.000 | 27.250 | 21.000 | 0.242 | 0.169 | 0.138 |
| ml | 26.432 | 20.201 | 10.055 | 0.212 | 0.156 | 0.125 |
| correlogram | 37.250 | 24.750 | 21.750 | 0.281 | 0.184 | 0.145 |
| general_relative | 37.250 | 27.500 | 23.500 | 0.269 | 0.179 | 0.140 |
| pairwise_relative | 35.750 | 31.500 | 22.000 | 0.267 | 0.182 | 0.139 |
| log | 36.750 | 31.000 | 21.250 | 0.272 | 0.185 | 0.139 |
| rodogram | 34.250 | 26.500 | 22.250 | 0.264 | 0.180 | 0.159 |
| cumulative | 32.500 | 28.250 | 12.750 | 0.263 | 0.189 | 0.144 |
| normal_score | 38.750 | 27.500 | 21.500 | 0.266 | 0.167 | 0.141 |
| truth_variogram | 7.500 | 7.500 | 7.500 | 0.070 | 0.070 | 0.070 |
| idw | nan | nan | nan | nan | nan | nan |

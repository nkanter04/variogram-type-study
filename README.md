# Variogram-type study, v2

Which experimental variogram type (GSLIB `gamv` 1, 4, 5, 6, 7, 8, plus cumulative and normal-score
semivariograms) gives the best continuity model, map and development decision from sparse wells?
And how does that compare with drilling more appraisal wells?

## Run

```bash
pip install -r requirements.txt
python tests/check_core.py
python run_study.py pilot
python run_study.py main
```

The results of both runs are already committed in `results/`, so the notebooks open without re-running the study.
Open them in order; each runs in under a minute:

1. `01_truths_and_wells.ipynb`: scenarios, truth models, exhaustive variograms, well designs, declustering
2. `02_variograms.ipynb`: experimental variograms, azimuth inference, fitting, pilot trimming rule, accuracy
3. `03_kriging_decisions.ipynb`: kriging, OOIP, uncertainty, well placement, value of information, executive summary

`tests/check_core.py` also compares the experimental variograms with the original study's saved results
when the environment variable `VARIOGRAM_ORIGINAL_DATA` points at that study's `data` folder.

Run times on a 16-core laptop: checks ~1 min, pilot ~40 s, main ~4 min. Needs numpy, scipy, pandas and matplotlib
(Anaconda); no GeostatsPy or GSLIB executables.

## Design

| | original study | v2 | why |
|---|---|---|---|
| truth simulation | GeostatsPy SGSIM | FFT circulant embedding | exact covariance, fast enough for 30 realizations |
| scenarios | 6 | gaussian, skewed, anisotropic, facies | each targets one variogram-type theory; homogeneous was trivially easy, short range is now a well-density question |
| realizations | 3 per scenario | 30 per scenario | scenario-level conclusions become testable |
| wells | 36; regular / random / preferential | 25, 50, 100; random / preferential | well count is the main lever; the regular grid had no short lags |
| lags | fixed 50 m | median nearest-neighbour spacing | first lags always populated |
| azimuth | given | inferred from the wells (variogram-map reading) | realistic |
| declustering | none | cell declustering (GSLIB declus rule) | preferential wells bias mean, variance and normal scores |
| types | 8 | 8 from the course list + normal score; trimming rule tested in a pilot | professor's list plus the practitioner baseline |
| model choice | — | leave-one-out cross-validation pick | can the best type be chosen from the wells? |
| benchmarks | truth variogram, naive, random | + inverse distance (no variogram) | separates value of data, of a map, of a variogram |
| uncertainty | kriging variance | + 50 conditional simulations (normal-score model) | P10-P90 OOIP coverage |
| decision | regret on kriged map | + expected-value placement (E-type by quadrature) | Lecture 16c |
| metrics | ~12 + go/no-go | primary: regret in $MM | one pre-stated endpoint |
| statistics | paired t, 54 datasets as independent | paired by truth model, Holm correction, 95% CIs | correct unit of replication |

## Code (`vstudy/`)

| module | role |
|---|---|
| `config.py` | every setting: grid, OOIP, scenarios, well counts, types, fitting grid, economics |
| `simulate.py` | FFT Gaussian fields, truth models (seeded per stage/scenario/realization) |
| `wells.py` | sampling designs, cell declustering, normal-score transform |
| `variography.py` | 8 measures on 12 azimuths, adaptive lags, exhaustive FFT truth variogram |
| `fitting.py` | azimuth inference + pair-weighted grid-search fit |
| `crossval.py` | closed-form leave-one-out kriging errors |
| `kriging.py` | ordinary/simple kriging, inverse distance, conditional simulation, E-type |
| `decisions.py` | drainage value, greedy + swap well placement, random benchmark |
| `pipeline.py` | per-truth workflow, parallel runner, cache; `build_dataset()` recreates any dataset |
| `stats.py`, `analysis.py`, `plots.py` | clustered tests, summary tables, figures |

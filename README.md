# Variogram-type study

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

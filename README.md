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
python tests/check_sgs.py
python run_study.py aoi
```

The results of all runs are already committed in `results/`, so the notebooks open without re-running the study.
Open them in order; each runs in under a minute:

1. `01_truths_and_wells.ipynb`: scenarios, truth models, exhaustive variograms, well designs, declustering
2. `02_variograms.ipynb`: experimental variograms, azimuth inference, fitting, pilot trimming rule, accuracy
3. `03_kriging_decisions.ipynb`: estimation by kriging and by sequential Gaussian simulation (50 realizations per
   variogram model), OOIP, uncertainty, well placement, value of information, executive summary
4. `04_aoi_simulation.ipynb`: area-of-interest volumetrics with sequential Gaussian simulation; how variogram azimuth,
   range and type shift the AOI average porosity and OOIP (global vs local AOIs, 3 realizations shown, 100 per model)

`tests/check_core.py` also compares the experimental variograms with the original study's saved results
when the environment variable `VARIOGRAM_ORIGINAL_DATA` points at that study's `data` folder.

Run times on a 16-core laptop: checks ~1 min each, pilot ~1 min, main ~40 min (mostly the simulations), aoi ~2 min. Needs numpy, scipy, pandas,
matplotlib and numba (all in Anaconda); no GeostatsPy or GSLIB executables.

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
| `sgs.py` | sequential Gaussian simulation (GSLIB sgsim algorithm, numba-compiled, ~0.4 s per realization) |
| `aoi.py` | AOIs as NumPy slices, AOI selection, SGS runner and figures for notebook 4 |
| `decisions.py` | drainage value, greedy + swap well placement, random benchmark |
| `pipeline.py` | per-truth workflow, parallel runner, cache; `build_dataset()` recreates any dataset |
| `stats.py`, `analysis.py`, `plots.py` | clustered tests, summary tables, figures |

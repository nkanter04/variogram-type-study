# Variogram-type study

Which experimental variogram type (GSLIB `gamv` 1, 4, 5, 6, 7, 8, plus cumulative and normal-score
semivariograms) gives the best continuity model, map and development decision from sparse wells?
And how does that compare with drilling more appraisal wells?

Synthetic truth models make every estimate and decision exactly scoreable: 4 reservoir scenarios x 30 truth models x
25 / 50 / 100 wells x random / preferential drilling = 720 datasets, each with 8 variogram types, kriging and sequential
Gaussian simulation, and two development decisions (where to place 3 wells; which area of interest to develop).

## Findings

**Bottom line: the choice of variogram type barely matters for the decision. What matters is how much unbiased well
data you have, how the map is used (a smooth kriged map fails once a cutoff is involved), and getting the variogram's
direction and range right, which sparse wells cannot do whatever the type.**

1. **No variogram type beats the semivariogram.** At 25, 50 and 100 wells none of the 7 alternatives makes significantly
   better or worse well-placement decisions; every difference is within about ±$0.27MM per 3-well program.
   [Notebook 3](03_kriging_decisions.ipynb), *The development decision*.
2. **A pre-registered test on fresh data confirms it.** Only picking the type by cross-validation at 25 wells was
   confirmed, and its gain (−$0.11MM) is below the $0.25MM pre-set threshold for mattering. Robust types gave no advantage
   even under outliers, a proportional effect or facies (the facies indicator variogram included).
   [confirmatory/results/RESULTS.md](confirmatory/results/RESULTS.md), hypotheses fixed in
   [confirmatory/hypotheses.json](confirmatory/hypotheses.json) before any data was generated.
3. **The variogram itself does matter.** Kriging with the true variogram is worth about $0.47MM per program at 50 wells,
   but sparse wells cannot recover it: the inferred azimuth is off by 20-38 degrees on average and many minor ranges hit
   the minimum. [Notebook 2](02_variograms.ipynb).
4. **Data is the lever.** Going from 25 to 50 wells is worth about $1.3MM and from 50 to 100 about $0.8MM; any map from the
   wells beats random drilling by about $8.2MM; an inverse-distance map with no variogram does as well as kriging at
   25-50 wells. [Notebook 3](03_kriging_decisions.ipynb), *Value of information*.
5. **Simulation (SGS) shows the uncertainty, not a better decision.** On realizations the types almost never pick the same
   wells (7% of datasets vs 24% on kriged maps), yet none is better, and deciding on realizations does not beat the kriged
   map because drained OOIP is a linear sum. Every fitted variogram makes the P10-P90 too narrow; with sweet-spot
   (preferential) wells the OOIP range misses the truth 95-99% of the time.
   [Notebook 3](03_kriging_decisions.ipynb), *Estimation by sequential Gaussian simulation*.
6. **Areas of interest: the variogram matters locally, but rarely changes the choice.** With AOIs fixed before modelling,
   the types disagree on OOIP by about 3% over the whole field, 9% over a 400 m block and 15% over a 200 m block. Still,
   they pick the same 200 m block to develop as the semivariogram about 9 times in 10, and only the true variogram is
   significantly better (+$0.05-0.07MM). [Notebook 3](03_kriging_decisions.ipynb), *Volumetrics in the areas of interest*.
   In a hand-picked sensitive area, a 90-degree azimuth error cuts the AOI OOIP by 24% and flips the decision
   ([notebook 4](04_aoi_simulation.ipynb), teaching demonstration).
7. **With a net-pay cutoff, kriging's smoothing costs real money.** When only rock above a porosity cutoff counts, placing
   wells on the expected net OOIP (the cutoff applied to each cell's uncertainty, which is what averaging SGS realizations
   converges to) beats placing them on the kriged map by $0.51MM per program (pre-registered, 400 fresh truth models,
   confirmed). With random wells the kriged map understates field net OOIP by about 45%; the expected value is unbiased.
   Even so, 6 of the 7 alternative variogram types remain equivalent to the semivariogram; the cumulative variogram is
   slightly better (−$0.12MM, below the $0.25MM threshold). [Notebook 5](05_netpay_and_likelihood.ipynb),
   [netpay/results/RESULTS.md](netpay/results/RESULTS.md).
8. **A better fitting method recovers the variogram better, but only pays off with many wells.** Maximum likelihood
   (no experimental variogram) halves the azimuth error at 100 wells (11 vs 21 degrees), yet pooled over well counts its
   decisions are equivalent to the semivariogram's. Split by well count (exploratory): it is worse at 25 wells, even at 50
   and better at 100. Constraining the automatic fit helps a little (−$0.03MM, confirmed) but kriging still does not beat
   inverse distance at 25-50 wells. [Notebook 5](05_netpay_and_likelihood.ipynb).

| Lever | Value per 3-well program |
|---|---|
| Any map from well data vs random drilling | about $8.2MM |
| 25 → 50 appraisal wells | about $1.3MM |
| 50 → 100 appraisal wells | about $0.8MM |
| Knowing the true variogram (50 wells) | about $0.5MM |
| Net-pay decision: expected net OOIP instead of the kriged map | about $0.5MM |
| Best variogram type, or cross-validation pick | about $0.1MM, not significant |

**Recommendation.** Use the semivariogram (or pick the type by cross-validation) and do not spend effort comparing
types. Whenever a cutoff or threshold is involved, decide on expected values from simulation, never on the kriged map.
Spend on more appraisal wells drilled without targeting sweet spots, and treat uncertainty ranges built from
preferentially sampled wells with caution.

## Run

```bash
pip install -r requirements.txt
python tests/check_core.py
python tests/check_sgs.py
python run_study.py pilot
python run_study.py main
python run_study.py aoi
python confirmatory/run_confirmatory.py
python confirmatory/analyze.py
python netpay/run_netpay.py
python netpay/analyze.py
```

The results of all runs are already committed in `results/`, `confirmatory/results/` and `netpay/results/`, so the notebooks open without
re-running anything. Open them in order; each runs in about a minute:

1. `01_truths_and_wells.ipynb`: scenarios, truth models, exhaustive variograms, areas of interest (fixed before
   modelling), well designs, declustering
2. `02_variograms.ipynb`: experimental variograms, azimuth inference, fitting, pilot trimming rule, accuracy
3. `03_kriging_decisions.ipynb`: estimation by kriging and by sequential Gaussian simulation (50 realizations per
   variogram model), OOIP, uncertainty, well placement, AOI volumetrics and AOI decision, value of information,
   executive summary
4. `04_aoi_simulation.ipynb`: teaching demonstration on one case of when the variogram matters for AOI volumetrics
   (correct vs rotated azimuth, 3 SGS realizations shown, 100 per model)

5. `05_netpay_and_likelihood.ipynb`: pre-registered test of a net-pay cutoff decision (kriged map vs expected net OOIP)
   and of maximum-likelihood and constrained variogram fitting

`confirmatory/` and `netpay/` are the two pre-registered tests (see their READMEs); they only import `vstudy`.

`tests/check_core.py` also compares the experimental variograms with the original study's saved results
when the environment variable `VARIOGRAM_ORIGINAL_DATA` points at that study's `data` folder.

Run times on a 16-core laptop: checks ~1-2 min each, pilot ~1 min, main ~40 min (mostly the simulations), aoi ~1 min,
confirmatory ~12 min, netpay ~35 min. Needs numpy, scipy, pandas, matplotlib and numba (all in Anaconda); no GeostatsPy or GSLIB
executables. Keep the computer awake during the long runs: they pause while it sleeps.

## Code (`vstudy/`)

| module | role |
|---|---|
| `config.py` | every setting: grid, OOIP, scenarios, well counts, types, fitting grid, AOIs, economics |
| `simulate.py` | FFT Gaussian fields, truth models (seeded per stage/scenario/realization) |
| `wells.py` | sampling designs, cell declustering, normal-score transform |
| `variography.py` | 8 measures on 12 azimuths, adaptive lags, exhaustive FFT truth variogram |
| `fitting.py` | azimuth inference + pair-weighted grid-search fit |
| `crossval.py` | closed-form leave-one-out kriging errors |
| `kriging.py` | ordinary/simple kriging, inverse distance, FFT conditional simulation, E-type |
| `sgs.py` | sequential Gaussian simulation (GSLIB sgsim algorithm, numba-compiled, ~0.05 s per realization) |
| `areas.py` | areas of interest as NumPy slices; the fixed study AOIs |
| `aoi.py` | single-case AOI demonstration for notebook 4: AOI selection, SGS runner, figures |
| `decisions.py` | drainage value, greedy + swap well placement, risk-averse (P10) placement, random benchmark |
| `pipeline.py` | per-truth workflow, parallel runner, cache; `build_dataset()` recreates any dataset |
| `stats.py`, `analysis.py`, `plots.py` | clustered tests, summary tables, figures |

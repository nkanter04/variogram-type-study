# CLAUDE.md

PGE 338 (UT Austin, Prof. Pyrcz) student project: does the experimental variogram type change kriged porosity, OOIP and
development decisions? Synthetic truth models make every estimate exactly scoreable. Findings are in README.md.

## Running

- Use the Anaconda Python (`%USERPROFILE%\anaconda3\python.exe` on the author's Windows machine). The other `python`
  on PATH (3.14) is not the environment the results were produced with (numba, pandas versions).
- Runs that use multiprocessing (`run_study.py`, `confirmatory/run_confirmatory.py`) must be started from a script file,
  not piped through stdin: Windows spawn re-imports `__main__` and fails on `<stdin>`.
- `python run_study.py main` takes ~40 min (mostly SGS). It pauses while the laptop sleeps; keep the machine awake and
  plugged in, run it in the background, and expect results only when all 120 truth models finish (written at the end).
- Checks: `tests/check_core.py` (~2 min) and `tests/check_sgs.py` (~1 min). Run both after changing anything in `vstudy/`.
- Notebooks only read cached results (`results/`, `confirmatory/results/`). Execute them headless with
  `python -m nbconvert --to notebook --execute --inplace <nb>.ipynb`; edit notebooks programmatically with nbformat.

## Rules that keep results valid

- **Results must stay reproducible.** Every random draw comes from a generator keyed by stage/scenario/realization/
  well count/design (`pipeline.dataset_rng`, `simulate.truth_rng`). New random steps must use their own keyed stream
  (as `simulate_and_decide` does), never the shared dataset `rng`, or all downstream results change. After a re-run,
  diff the existing CSVs against `git show HEAD:results/main/...` and confirm they are bit-identical.
- **`confirmatory/hypotheses.json` and `netpay/hypotheses.json` are pre-registrations.** Never edit them or re-run those studies with a
  changed design and present the result as confirmatory; any new hypothesis needs a new pre-registered file and fresh
  seeds, committed before data is generated.
- **AOIs are fixed before modelling** (`config.AOIS`, `vstudy/areas.py`). Never choose study AOIs from data or results;
  notebook 4 picks AOIs after kriging only as a labelled teaching demonstration.
- Statistics compare methods per truth model (the unit of replication), with Holm correction. Do not report naive
  t-tests over datasets as significance.

## Conventions

- GSLIB conventions: azimuth clockwise from north; `it` 1 = spherical, 2 = exponential; 100 x 100 grid of 10 m cells,
  cell centres at 5 m; **grid row 0 is the north edge**.
- All settings live in `vstudy/config.py`; notebooks call `vstudy` functions and only plot.
- The decision metric is regret in $MM per 3-well program (RF 0.30, $35/bbl netback).
- `netpay/` (branch `netpay-ml` until merged) adds a net-pay cutoff decision, maximum-likelihood and constrained fits;
  `netpay/pilot_results/` is exploratory only.

## Gotchas

- `vstudy/sgs.py` is a numba-compiled GSLIB sgsim (~0.05 s per realization). GeostatsPy's `sgsim` takes ~95 s per
  realization here and fails with a singular matrix when data sit exactly on grid-node centres (all wells here do);
  offset coordinates by ~1 m if it must be used.
- Pushing large result files can fail with "remote end hung up"; this repo sets `http.postBuffer` to 500 MB, and pushing
  one branch at a time helps.
- The repository is **public**: no personal paths, emails or local data locations in committed files (the original
  study's data folder is read from the `VARIOGRAM_ORIGINAL_DATA` environment variable).
- Commit messages end with a `Co-Authored-By: Claude` line; the user also edits the README directly on GitHub, so pull
  before editing it.

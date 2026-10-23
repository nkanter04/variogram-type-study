# Confirmatory run (separate add-on)

The main study found no variogram type that significantly beats the semivariogram, but left a few
suggestive leads. This folder tests those leads, plus the conditions robust variograms were designed
for, as a pre-registered confirmatory experiment.

- `hypotheses.json`: the hypotheses, scenarios, sample size, test and decision rule. It was committed
  before any confirmatory data was generated; the git history is the proof.
- `scenarios.py`: the two new scenarios (outliers, proportional effect) and indicator variograms.
- `run_confirmatory.py`: generates fresh truth models on a seed the main study never used and saves
  `results/`.
- `analyze.py`: applies the pre-stated test and writes `RESULTS.md`.

```bash
python confirmatory/run_confirmatory.py
python confirmatory/analyze.py
```

## Separation from the main workflow

This folder only *imports* the `vstudy` package; it changes nothing in `vstudy/`, the notebooks or
`results/`. It lives on the `confirmatory` git branch, so `main` is untouched.

To remove it completely:

```bash
git checkout main
git branch -D confirmatory
git push origin --delete confirmatory
```

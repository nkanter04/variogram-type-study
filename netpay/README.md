# Net-pay decision and maximum-likelihood variograms (separate add-on)

Two limits of the main study are tested here as a pre-registered experiment:

1. **The decisions were sums.** Placing wells or choosing a block by total OOIP is close to linear, and for a sum the
   smooth kriged map is already near the best answer, so simulation and the variogram type had little room to matter.
   Here only rock above a **net-pay cutoff** counts. A smoothed map misjudges how much rock passes a cutoff; the expected
   value over each cell's local distribution (what averaging SGS realizations converges to) does not.
2. **Every variogram type passed through the same automatic fit.** Two alternatives separate "type" from "fitting":
   the semivariogram fitted with **resolvability constraints** (no range below the well spacing, anisotropy at most 5:1),
   and **maximum likelihood**, which infers the model straight from the wells with no experimental variogram.

Files:

- `hypotheses.json`: 13 hypotheses, definitions, sample size, test and decision rule; committed before any data on its
  seed was generated.
- `core.py`: cutoff, expected net OOIP, constrained fit, maximum likelihood.
- `run_netpay.py`: runner (`--pilot` for the exploratory pilot on a different seed).
- `analyze.py`: applies the pre-stated tests and writes `results/RESULTS.md`.
- `pilot_results/`: the exploratory pilot (40 truth models) used to choose the design. Not findings.

```bash
python netpay/run_netpay.py     # ~80 min on 14 processes; keep the computer awake
python netpay/analyze.py
```

This folder only imports `vstudy`; deleting it removes the add-on without affecting anything else.

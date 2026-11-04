"""New scenarios and indicator variograms for the confirmatory run (see hypotheses.json).

Only imports vstudy; nothing in vstudy is modified."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vstudy import config as C, fitting                                 # noqa: E402
from vstudy.simulate import gaussian_fields, make_truth                  # noqa: E402
from vstudy.variography import experimental                              # noqa: E402

SCENARIOS = ['gaussian', 'skewed', 'anisotropic', 'facies', 'outliers', 'proportional']
WELLS = {'gaussian': [25], 'skewed': [25, 50], 'anisotropic': [25], 'facies': [25, 50],
         'outliers': [50], 'proportional': [50]}                         # only what H1-H7 need
OUTLIER_FRACTION, OUTLIER_SD = 0.10, 0.10


def _std_field(m, rng):
    z = gaussian_fields(m, 1, rng)[0]
    return (z - z.mean()) / z.std()


def truth(scenario, rng):
    """Porosity grid and facies grid (or None)."""
    if scenario in C.SCENARIOS:
        return make_truth(scenario, rng)
    if scenario == 'outliers':                       # same truth as 'gaussian'; errors are added at the wells
        return make_truth('gaussian', rng)
    if scenario == 'proportional':                   # local std proportional to the local mean
        z1 = _std_field(dict(it=1, nug=0.0, azi=0.0, hmaj=800.0, hmin=800.0), rng)
        y = _std_field(dict(it=1, nug=0.1, azi=0.0, hmaj=300.0, hmin=300.0), rng)
        m = np.clip(0.16 + 0.05 * z1, 0.05, 0.30)
        return np.clip(m * (1.0 + 0.35 * y), C.PHI_LO, C.PHI_HI), None
    raise ValueError(scenario)


def recorded_values(scenario, true_v, rng):
    """Porosity as recorded at the wells: gross errors on 10% of wells in the outlier scenario."""
    v = np.asarray(true_v, float).copy()
    if scenario == 'outliers':
        k = int(round(OUTLIER_FRACTION * len(v)))
        idx = rng.choice(len(v), size=k, replace=False)
        v[idx] = np.clip(v[idx] + rng.normal(0.0, OUTLIER_SD, k), C.PHI_LO, C.PHI_HI)
    return v


def indicator_model(x, y, ind):
    """Fitted model of an indicator semivariogram (gamv 9 or 10), used as the continuity model.

    The indicator is shifted by +1 so the shared experimental() routine never takes log(0); the
    semivariogram is unchanged by the shift."""
    ind = np.asarray(ind, float)
    if ind.min() == ind.max():                       # all wells in one class: no spatial information
        return dict(it=1, nug=1.0, azi=0.0, hmaj=C.FIT_RANGES[0], hmin=C.FIT_RANGES[0])
    e = experimental(x, y, ind + 1.0, np.zeros(len(ind)))
    m = fitting.fit(e[e.vtype == 'semivariogram'])
    return {k: m[k] for k in ('it', 'nug', 'azi', 'hmaj', 'hmin')}

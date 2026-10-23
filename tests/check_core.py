"""Core checks. Run:  python tests/check_core.py   (prints PASS/FAIL per check)."""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vstudy import config as C, crossval, fitting, wells as W            # noqa: E402
from vstudy.models import gamma_along                                   # noqa: E402
from vstudy.simulate import gaussian_fields, make_truth                  # noqa: E402
from vstudy.variography import Truth, experimental, variogram_map        # noqa: E402
from vstudy.pipeline import run_truth                                    # noqa: E402

# optional: the original 4-notebook study's data folder, to compare experimental variograms against
ORIGINAL = os.environ.get('VARIOGRAM_ORIGINAL_DATA', '')
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


# 1. gamv conventions: reproduce the original study's experimental variograms (original lag settings)
if ORIGINAL and os.path.exists(os.path.join(ORIGINAL, 'variograms', 'experimental.csv')):
    ex = pd.read_csv(os.path.join(ORIGINAL, 'variograms', 'experimental.csv'))
    azis = {'homogeneous': 0, 'heterogeneous': 0, 'anisotropic': 45, 'skewed': 0, 'trend': 0, 'facies': -60}
    worst, bad_n = 0.0, 0
    for (s, r, d), theirs in ex.groupby(['scenario', 'realization', 'design']):
        w = pd.read_csv(os.path.join(ORIGINAL, 'wells', f'{s}_r{r}_{d}.csv'))
        a = azis[s]
        mine = experimental(w.X.values, w.Y.values, w.Por.values, np.zeros(len(w)),
                            lags=dict(xlag=50.0, xltol=25.0, nlag=12, bandwh=175.0), azimuths=[a, a + 90.0])
        mine['direction'] = np.where(mine.azi == a, 'major', 'minor')
        m = mine.merge(theirs, on=['vtype', 'direction', 'lag'], suffixes=('', '_o'))
        m = m[m.vtype.isin(['semivariogram', 'correlogram', 'general_relative', 'pairwise_relative', 'log', 'rodogram', 'cumulative'])]
        worst = max(worst, float(np.nanmax(np.abs(m.std_value - m.std_value_o))))
        bad_n += int((m.npairs != m.npairs_o).sum())
    check('experimental variograms = original study', worst < 1e-9 and bad_n == 0, f'max diff {worst:.1e}, pair-count mismatches {bad_n}')

# 2. simulator reproduces its input model
rng = np.random.default_rng(1)
m = C.SCENARIOS['anisotropic']['vario']
g = np.mean([variogram_map(z) for z in gaussian_fields(m, 40, rng)], 0) / 1.0
tr = Truth.__new__(Truth); tr.map = g
cur = tr.curves(h=np.array([50.0, 100.0, 200.0, 400.0]), azimuths=[45.0, 135.0])
err = max(np.abs(cur[cur.azi == a].std_value.values - gamma_along(np.array([50.0, 100.0, 200.0, 400.0]), a, m)).max() for a in (45.0, 135.0))
check('FFT simulation reproduces the input variogram', err < 0.05, f'max |gamma error| over 40 fields = {err:.3f}')

# 3. closed-form leave-one-out = brute force
x, y, v = rng.uniform(0, 1000, 30), rng.uniform(0, 1000, 30), rng.normal(size=30)
mod = dict(it=1, nug=0.1, azi=30.0, hmaj=400.0, hmin=200.0)
fast = crossval.loo_errors(x, y, v, mod)
slow = []
for i in range(30):
    k = np.arange(30) != i
    from vstudy.kriging import weights
    lam, _ = weights(x[k], y[k], mod, targets=np.array([[x[i], y[i]]]))
    slow.append(v[i] - (lam[:, 0] @ v[k]))
check('closed-form LOOCV = brute force', np.allclose(fast, slow, atol=1e-10), f'max diff {np.abs(fast - np.array(slow)).max():.1e}')

# 4. declustering removes most of the preferential-sampling bias in the mean
bias_raw, bias_dec = [], []
for k in range(40):
    phi, _ = make_truth('skewed', np.random.default_rng([5, k]))
    w = W.sample(phi, 'preferential', 50, np.random.default_rng([6, k]))
    wt = W.decluster(w['x'], w['y'], w['v'], preferential=True)
    bias_raw.append(w['v'].mean() - phi.mean()); bias_dec.append(np.sum(wt * w['v']) - phi.mean())
check('declustering reduces preferential mean bias', abs(np.mean(bias_dec)) < abs(np.mean(bias_raw)),
      f'mean bias raw {np.mean(bias_raw):+.4f}, declustered {np.mean(bias_dec):+.4f} ({100 * (1 - np.mean(bias_dec) / np.mean(bias_raw)):.0f}% removed)')

# 5. azimuth inference recovers the true direction from the exhaustive variogram
phi, _ = make_truth('anisotropic', np.random.default_rng(11))
ref = fitting.fit(Truth(phi).curves())
check('azimuth inferred from exhaustive anisotropic truth', fitting.azimuth_error(ref, 45.0) <= 15.0,
      f"fitted azi {ref['azi']:.0f} (true 45), hmaj {ref['hmaj']:.0f}, hmin {ref['hmin']:.0f}")

# 6. E-type map by quadrature = average of many conditional simulations
from vstudy import kriging                                               # noqa: E402
phi, _ = make_truth('skewed', np.random.default_rng(21))
w = W.sample(phi, 'random', 50, np.random.default_rng(22))
ns = W.NormalScore(w['v'], np.full(50, 1 / 50))
m = dict(it=1, nug=0.1, azi=0.0, hmaj=400.0, hmin=250.0)
et = kriging.expected_ooip_cells(w['x'], w['y'], ns, m).sum()
sims = kriging.conditional_sims(w['x'], w['y'], ns, m, 400, np.random.default_rng(23))
mc = C.ooip_cells(sims).sum((1, 2))
check('E-type quadrature = mean of 400 simulations (total OOIP)', abs(et - mc.mean()) < 3 * mc.std() / np.sqrt(400),
      f'quadrature {et / 1e6:.4f} vs simulations {mc.mean() / 1e6:.4f} +/- {mc.std() / 1e6 / 20:.4f} MMSTB')

# 7. one full truth model runs, with timing
t0 = time.time()
out = run_truth(('pilot', 'facies', 0))
d = pd.DataFrame(out['decisions'])
check('one truth model end to end', len(d) == len(C.WELL_COUNTS) * len(C.DESIGNS) * (len(C.VTYPES) + 5),
      f'{len(d)} decisions in {time.time() - t0:.1f} s')

print(f"\n{sum(results)}/{len(results)} checks passed")

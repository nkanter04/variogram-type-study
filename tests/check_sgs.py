"""Checks for the compiled sequential Gaussian simulation. Run:  python tests/check_sgs.py"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vstudy import config as C, kriging, sgs, wells as W               # noqa: E402
from vstudy.models import gamma_along                                  # noqa: E402
from vstudy.pipeline import build_truth, dataset_rng                   # noqa: E402
from vstudy.variography import Truth, variogram_map                    # noqa: E402

results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


m = dict(it=1, nug=0.05, azi=45.0, hmaj=600.0, hmin=150.0)
none = np.array([])
sgs.simulate_scores(none, none, none, m, 1, np.random.default_rng(0))         # compile

# 1. unconditional realizations reproduce the variogram model (major and minor directions)
t0 = time.time()
u = sgs.simulate_scores(none, none, none, m, 20, np.random.default_rng(1))
tr = Truth.__new__(Truth); tr.map = np.mean([variogram_map(z) for z in u], 0)
h = np.array([50.0, 100.0, 200.0, 400.0])
err = max(np.abs(tr.curves(h=h, azimuths=[a]).std_value.values - gamma_along(h, a, m)).max() for a in (45.0, 135.0))
check('SGS reproduces the variogram model', err < 0.08, f'max |gamma error| over 20 realizations = {err:.3f} ({(time.time() - t0) / 20:.2f} s per realization)')

# 2. conditional realizations honour the wells exactly
phi = build_truth('main', 'anisotropic', 0)[0]
w = W.sample(phi, 'random', 25, dataset_rng('main', 'anisotropic', 0, 25, 'random'))
ns = W.NormalScore(w['v'], np.full(25, 1 / 25))
ys = sgs.simulate_scores(w['x'], w['y'], ns.scores, m, 100, np.random.default_rng(2))
e = np.abs(ys[:, w['row'], w['col']] - ns.scores[None, :]).max()
check('wells honoured', e == 0.0, f'max |simulated - well normal score| = {e:.1e}')

# 3. the average of many realizations converges to simple kriging
lam, _ = kriging.weights(w['x'], w['y'], m, ok=False)
r = np.corrcoef(ys.mean(0).ravel(), (lam.T @ ns.scores))[0, 1]
check('mean of 100 realizations ~ simple kriging', r > 0.97, f'correlation {r:.3f}')

# 4. the spread of the field average matches the exact FFT conditional simulation
p = ns.back(ys).mean((1, 2)).std()
f = kriging.conditional_sims(w['x'], w['y'], ns, m, 100, np.random.default_rng(3)).mean((1, 2)).std()
check('std of the field-average porosity: SGS vs exact FFT simulation', 0.7 < p / f < 1.4, f'SGS {p:.5f}, FFT {f:.5f}')

print(f"\n{sum(results)}/{len(results)} checks passed")

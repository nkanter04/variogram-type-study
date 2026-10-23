"""Appraisal wells: sampling designs, cell declustering and the normal-score transform."""
import numpy as np
from scipy.stats import norm

from . import config as C


def sample(phi, design, n, rng):
    """Well locations and porosity. random = uniform; preferential = weighted by porosity rank
    (drilling the sweet spots, weights WT_MIN..WT_MAX)."""
    if design == 'random':
        p = None
    elif design == 'preferential':
        rank = phi.ravel().argsort().argsort() / (phi.size - 1)
        w = C.WT_MIN + (C.WT_MAX - C.WT_MIN) * rank
        p = w / w.sum()
    else:
        raise ValueError(design)
    idx = rng.choice(phi.size, size=n, replace=False, p=p)
    rr, cc = np.unravel_index(idx, phi.shape)
    return dict(x=C.XC[cc], y=C.YC[rr], v=phi[rr, cc], row=rr, col=cc)


def _cell_weights(x, y, cell, n_offsets, rng):
    w = np.zeros(len(x))
    for ox, oy in rng.uniform(0, cell, (n_offsets, 2)):
        key = np.floor((x - C.XMIN + ox) / cell).astype(int) * 100000 + np.floor((y - C.YMIN + oy) / cell).astype(int)
        _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        w += 1.0 / counts[inv] / len(counts)
    return w / w.sum()


def decluster(x, y, v=None, preferential=False, n_offsets=16):
    """Cell declustering weights (sum to 1), averaged over several grid origins as in GSLIB declus.

    preferential=True (wells known to target sweet spots): the cell size between half the nominal
    spacing and 400 m that minimizes the declustered mean (GSLIB declus rule for high-biased data).
    Otherwise the nominal well spacing sqrt(area / n), which leaves evenly spread data near equal weight."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    spacing = np.sqrt((C.XMAX - C.XMIN) * (C.YMAX - C.YMIN) / len(x))
    if not preferential:
        return _cell_weights(x, y, spacing, n_offsets, np.random.default_rng(0))
    best = None
    for cell in np.arange(0.5 * spacing, 400.0 + 1e-9, 25.0):
        w = _cell_weights(x, y, cell, n_offsets, np.random.default_rng(0))
        m = float(np.sum(w * v))
        if best is None or m < best[0]:
            best = (m, w)
    return best[1]


def weighted_stats(v, w):
    m = float(np.sum(w * v))
    return m, float(np.sum(w * (v - m) ** 2))


class NormalScore:
    """Declustered normal-score transform with linear tails to the physical porosity limits."""

    def __init__(self, v, w):
        o = np.argsort(v)
        vs, ws = np.asarray(v, float)[o], np.asarray(w, float)[o]
        cdf = np.cumsum(ws) - 0.5 * ws                          # mid-point cumulative probability
        z = norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
        sd = vs.std()
        lo, hi = max(C.PHI_LO, vs[0] - 0.5 * sd), min(C.PHI_HI, vs[-1] + 0.5 * sd)
        self.z_tab, self.v_tab = np.r_[-4.0, z, 4.0], np.r_[lo, vs, hi]
        self.scores = np.empty(len(v)); self.scores[o] = z

    def back(self, y):
        return np.interp(y, self.z_tab, self.v_tab)

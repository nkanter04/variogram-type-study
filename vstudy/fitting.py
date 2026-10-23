"""Automatic variogram modelling, identical for every continuity measure.

The major direction is inferred from the wells: the azimuth of greatest continuity (lowest
short-lag values on the directional curves, as read from a variogram map). Then pair-weighted
least squares by exhaustive grid search over structure (spherical / exponential), nugget, major
range and minor range (h_min <= h_maj), sill 1.
"""
import numpy as np

from . import config as C
from .models import unit_structure

_MIN_LAGS = 3            # a direction needs at least this many usable lags to be fitted
AZI_HMAX = 300.0         # lags used to read the direction of greatest continuity


def _sse(h, y, w, cumulative):
    """SSE[structure, nugget, range] for one directional curve."""
    nugs, rng = C.FIT_NUGS[:, None, None], C.FIT_RANGES[None, :, None]
    dh = np.diff(np.r_[0.0, h])
    out = np.empty((len(C.FIT_STRUCTURES), len(C.FIT_NUGS), len(C.FIT_RANGES)))
    for s, it in enumerate(C.FIT_STRUCTURES):
        m = nugs + (1.0 - nugs) * unit_structure(it, h[None, None, :] / rng)
        if cumulative:
            m = np.cumsum(m * dh, axis=-1) / h
        out[s] = (w * (m - y) ** 2).sum(-1)
    return out


def _usable(curve):
    c = curve[(curve.npairs >= C.MIN_PAIRS)].dropna(subset=['std_value', 'h']).sort_values('h')
    return c.h.values.astype(float), c.std_value.values.astype(float), c.npairs.values.astype(float)


def major_direction(curves, hmax=AZI_HMAX):
    """Direction of greatest continuity: the azimuth whose curve is lowest on average over lags
    <= hmax (the variogram-map reading taught in PGE 338), smoothed over neighbouring azimuths.
    Returns candidate azimuths ordered best first."""
    score = {}
    for azi, c in curves.groupby('azi'):
        c = c[(c.npairs >= C.MIN_PAIRS) & (c.h <= hmax)].dropna(subset=['std_value'])
        if len(c):
            score[float(azi)] = np.average(c.std_value, weights=c.npairs)
    az = np.array(sorted(score))
    if len(az) == 0:
        return []
    s = np.array([score[a] for a in az])
    step = C.AZIMUTHS[1] - C.AZIMUTHS[0]
    sm = []
    for a, v in zip(az, s):                      # circular 1-2-1 smoothing over available neighbours
        nb = [score.get((a + d) % 180.0) for d in (-step, step)]
        nb = [x for x in nb if x is not None]
        sm.append((2 * v + sum(nb)) / (2 + len(nb)))
    return list(az[np.argsort(sm)])


def fit(curves, cumulative=False, azimuths=None):
    """curves: experimental rows of ONE measure for all azimuths (columns azi, h, std_value, npairs).

    The major azimuth comes from major_direction(); the model is then fitted to the major and minor
    curves. Returns dict(it, nug, hmaj, hmin, azi, wrmse)."""
    tabs, wsum = {}, {}
    for azi, c in curves.groupby('azi'):
        h, y, w = _usable(c)
        if len(h) >= _MIN_LAGS:
            tabs[float(azi)], wsum[float(azi)] = _sse(h, y, w, cumulative), w.sum()
    best = None
    for azi in (azimuths if azimuths is not None else major_direction(curves)):
        mnr = (azi + 90.0) % 180.0
        if azi not in tabs or mnr not in tabs:
            continue
        smaj, smin = tabs[azi], tabs[mnr]
        best_min = np.minimum.accumulate(smin, axis=2)                 # best h_min not above each h_maj
        arg = np.zeros(smin.shape, int)
        for k in range(1, smin.shape[2]):
            arg[:, :, k] = np.where(smin[:, :, k] < best_min[:, :, k - 1], k, arg[:, :, k - 1])
        total = smaj + best_min
        s, n, k = np.unravel_index(np.argmin(total), total.shape)
        wrmse = float(np.sqrt(total[s, n, k] / (wsum[azi] + wsum[mnr])))
        if best is None or wrmse < best['wrmse']:
            best = dict(it=int(C.FIT_STRUCTURES[s]), nug=float(C.FIT_NUGS[n]), hmaj=float(C.FIT_RANGES[k]),
                        hmin=float(C.FIT_RANGES[arg[s, n, k]]), azi=float(azi), wrmse=wrmse)
        if azimuths is None:
            break                                                      # first feasible = most continuous
    if best is None:                                                   # too few pairs anywhere: pure nugget
        best = dict(it=1, nug=1.0, hmaj=C.FIT_RANGES[0], hmin=C.FIT_RANGES[0], azi=0.0, wrmse=np.nan)
    return best


def fit_all(exp):
    """Fit every measure in an experimental table from variography.experimental."""
    return {vt: fit(exp[exp.vtype == vt], cumulative=(vt == 'cumulative')) for vt in C.VTYPES}


def azimuth_error(model, true_azi):
    """Angular error of the major direction, 0-90 deg (directions, not senses)."""
    d = abs((model['azi'] - true_azi) % 180.0)
    return float(min(d, 180.0 - d))

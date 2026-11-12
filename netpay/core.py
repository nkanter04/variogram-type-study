"""Net-pay decision, constrained fitting and maximum-likelihood variogram inference.

Self-contained add-on: imports vstudy, changes nothing in it.

Why. The main study's decisions sum OOIP over an area, and for a sum the smooth kriged map is
already near the best answer, so neither simulation nor the variogram type had much room to matter.
A net-pay cutoff makes the transfer function non-linear: only rock above the cutoff counts, and a
smoothed map misjudges how much rock that is. The automatic fit is also a suspect: all variogram
types pass through it, so it may hide them. Two alternatives are added: the same fit with
resolvability constraints, and maximum likelihood, which uses no experimental variogram at all.
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vstudy import config as C, fitting, kriging                       # noqa: E402
from vstudy.models import covariance                                    # noqa: E402

# ---------------------------------------------------------------- net-pay cutoff
NET_QUANTILE = 0.70           # only the best 30% of the rock (by the scenario's porosity distribution) is pay
N_Q = 100                     # equal-probability nodes for local expectations


def cutoff(scenario):
    """Porosity cutoff = NET_QUANTILE of the scenario's population distribution (known up front,
    like a regulatory or economic cutoff; never estimated from the wells)."""
    s = C.SCENARIOS[scenario]
    z = norm.ppf(NET_QUANTILE)
    if s['transform'] == 'gaussian':
        return s['mean'] + s['std'] * z
    if s['transform'] == 'lognormal':
        return s['median'] * np.exp(s['log_std'] * z)
    if s['transform'] == 'facies':                      # mixture: solve p_sand F_sand(c) + p_shale F_shale(c) = q
        grid = np.linspace(C.PHI_LO, C.PHI_HI, 4000)
        cdf = (s['sand_prop'] * norm.cdf(grid, *s['sand']) + (1 - s['sand_prop']) * norm.cdf(grid, *s['shale']))
        return float(grid[np.searchsorted(cdf, NET_QUANTILE)])
    raise ValueError(scenario)


def net_cells(phi, c):
    """Net OOIP per cell (STB): OOIP counts only where porosity reaches the cutoff."""
    phi = np.asarray(phi)
    return C.ooip_cells(phi) * (phi >= c)


def etype_net(x, y, ns, m, c):
    """Expected net OOIP per cell given the wells, by multi-Gaussian kriging.

    In normal-score space each cell is Gaussian (simple-kriging mean and variance). The expectation
    of the net-OOIP transfer function is the average over N_Q equal-probability values of that local
    distribution, back-transformed. This is what the average of infinitely many SGS realizations
    converges to, without Monte Carlo noise; it is exact here because drained net OOIP is a sum over cells."""
    lam, var = kriging.weights(np.asarray(x, float), np.asarray(y, float), m, ok=False)
    mu, sd = lam.T @ ns.scores, np.sqrt(var)
    t = norm.ppf((np.arange(N_Q) + 0.5) / N_Q)
    phi = ns.back(mu[:, None] + sd[:, None] * t[None, :])
    return net_cells(phi, c).mean(1).reshape(C.NY, C.NX)


# ---------------------------------------------------------------- constrained least-squares fit
MAX_RATIO = 5.0               # anisotropy h_maj / h_min a sparse well set can support


def fit_constrained(curves, min_range, max_ratio=MAX_RATIO):
    """vstudy.fitting.fit with two resolvability constraints: no range shorter than the well spacing
    (min_range; nothing closer than that is observed) and anisotropy ratio <= max_ratio."""
    R = C.FIT_RANGES
    for azi in fitting.major_direction(curves):
        tabs, wsum = [], 0.0
        for a in (azi, (azi + 90.0) % 180.0):
            h, yv, w = fitting._usable(curves[curves.azi == a])
            if len(h) < fitting._MIN_LAGS:
                break
            tabs.append(fitting._sse(h, yv, w, False)); wsum += w.sum()
        if len(tabs) < 2:
            continue
        smaj, smin = tabs
        best = None
        for k in np.where(R >= min_range)[0]:
            j = np.where((R >= max(min_range, R[k] / max_ratio)) & (R <= R[k]))[0]
            tot = smaj[:, :, k][:, :, None] + smin[:, :, j]
            s, n, jj = np.unravel_index(np.argmin(tot), tot.shape)
            if best is None or tot[s, n, jj] < best[0]:
                best = (tot[s, n, jj], s, n, k, j[jj])
        _, s, n, k, j = best
        return dict(it=int(C.FIT_STRUCTURES[s]), nug=float(C.FIT_NUGS[n]), hmaj=float(R[k]), hmin=float(R[j]),
                    azi=float(azi), wrmse=float(np.sqrt(best[0] / wsum)))
    return dict(it=1, nug=1.0, hmaj=float(R[0]), hmin=float(R[0]), azi=0.0, wrmse=np.nan)


# ---------------------------------------------------------------- maximum likelihood
ML_RANGES = np.array([50.0, 80.0, 125.0, 200.0, 300.0, 450.0, 700.0, 1000.0, 1500.0])
ML_NUGS = np.arange(0.0, 0.81, 0.1)


def _nll(z, dx, dy, m):
    """-2 log-likelihood (up to a constant) of z ~ N(mu 1, s2 R(m)), with mu and s2 profiled out."""
    try:
        L = np.linalg.cholesky(covariance(dx, dy, m) + 1e-8 * np.eye(len(z)))
    except np.linalg.LinAlgError:
        return np.inf
    a = np.linalg.solve(L, np.c_[z, np.ones(len(z))])
    mu = (a[:, 1] @ a[:, 0]) / (a[:, 1] @ a[:, 1])
    r = a[:, 0] - mu * a[:, 1]
    return len(z) * np.log(r @ r / len(z)) + 2.0 * np.log(np.diag(L)).sum()


def fit_ml(x, y, z):
    """Maximum-likelihood variogram model of the well normal scores z: no pairs, lags or experimental
    variogram. Coarse grid over structure, nugget, azimuth and ranges, then a local refinement."""
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
    best = (np.inf, None)
    for it in C.FIT_STRUCTURES:
        for nug in ML_NUGS:
            for i, hmaj in enumerate(ML_RANGES):
                for hmin in ML_RANGES[:i + 1]:
                    for azi in (C.AZIMUTHS[::2] if hmin < hmaj else C.AZIMUTHS[:1]):
                        m = dict(it=it, nug=nug, azi=azi, hmaj=hmaj, hmin=hmin)
                        v = _nll(z, dx, dy, m)
                        if v < best[0]:
                            best = (v, m)
    m0 = best[1]
    lo, hi = C.FIT_RANGES[0], C.FIT_RANGES[-1]

    def unpack(p):
        hmaj = float(np.clip(np.exp(p[1]), lo, hi))
        return dict(it=m0['it'], nug=float(np.clip(p[0], 0.0, 0.9)), azi=float(p[3] % 180.0), hmaj=hmaj,
                    hmin=float(np.clip(hmaj * np.exp(-abs(p[2])), lo, hmaj)))

    res = minimize(lambda p: _nll(z, dx, dy, unpack(p)),
                   [m0['nug'], np.log(m0['hmaj']), np.log(m0['hmaj'] / m0['hmin']), m0['azi']],
                   method='Nelder-Mead', options=dict(xatol=1e-2, fatol=1e-3, maxiter=300))
    m = unpack(res.x) if res.fun < best[0] else m0
    return dict(m, wrmse=np.nan)

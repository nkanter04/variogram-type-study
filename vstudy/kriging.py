"""Global-neighbourhood kriging, inverse distance, and conditional simulation."""
import numpy as np

from . import config as C
from .models import covariance
from .simulate import gaussian_fields


def weights(x, y, m, ok=True, targets=C.GRID_XY):
    """Kriging weights (n, ntarget) and kriging variance for a sill-1 model."""
    n = len(x)
    K = covariance(x[:, None] - x[None, :], y[:, None] - y[None, :], m)
    k = covariance(targets[None, :, 0] - x[:, None], targets[None, :, 1] - y[:, None], m)
    if ok:
        A = np.ones((n + 1, n + 1)); A[n, n] = 0.0; A[:n, :n] = K
        B = np.ones((n + 1, k.shape[1])); B[:n] = k
        sol = np.linalg.solve(A, B)
        lam, var = sol[:n], 1.0 - np.sum(sol[:n] * k, axis=0) - sol[n]
    else:
        lam = np.linalg.solve(K, k)
        var = 1.0 - np.sum(lam * k, axis=0)
    return lam, np.maximum(var, 0.0)


def ordinary_kriging(x, y, v, m, sill):
    """OK estimate and variance maps. The weights depend only on the model shape; sill (the
    declustered data variance) scales the variance."""
    lam, var = weights(np.asarray(x, float), np.asarray(y, float), m)
    est = np.clip(lam.T @ np.asarray(v, float), C.PHI_LO, C.PHI_HI)
    return est.reshape(C.NY, C.NX), (sill * var).reshape(C.NY, C.NX)


def idw(x, y, v, power=2.0):
    """Inverse-distance-squared map: a spatial estimate that uses no variogram."""
    d = np.hypot(C.GRID_XY[:, 0, None] - x[None, :], C.GRID_XY[:, 1, None] - y[None, :])
    w = 1.0 / np.maximum(d, 1e-6) ** power
    est = (w @ v) / w.sum(1)
    hit = d.min(1) < 1e-6
    est[hit] = v[d[hit].argmin(1)]
    return np.clip(est, C.PHI_LO, C.PHI_HI).reshape(C.NY, C.NX)


def conditional_sims(x, y, ns, m, n_sim, rng):
    """Gaussian conditional simulation in normal-score space, back-transformed to porosity.

    Unconditional FFT field + simple-kriging correction to the well normal scores (the conditioning
    step is exact for a Gaussian field, equivalent in distribution to sequential Gaussian simulation).
    ns: wells.NormalScore of the well data; m: variogram model of the normal scores."""
    lam, _ = weights(np.asarray(x, float), np.asarray(y, float), m, ok=False)
    rows = np.rint((C.YMAX - np.asarray(y)) / C.YSIZ - 0.5).astype(int)
    cols = np.rint((np.asarray(x) - C.XMIN) / C.XSIZ - 0.5).astype(int)
    u = gaussian_fields(m, n_sim, rng).reshape(n_sim, -1)
    ysim = u + (ns.scores[None, :] - u[:, rows * C.NX + cols]) @ lam
    return ns.back(ysim).reshape(n_sim, C.NY, C.NX)


def expected_ooip_cells(x, y, ns, m):
    """E[OOIP per cell] given the wells (the E-type map), without simulation.

    In normal-score space each cell is Gaussian with the simple-kriging mean and variance, so
    E[ooip(back(Y))] is a 1-D integral per cell, done with Gauss-Hermite quadrature. Because drained
    OOIP is a sum over cells, its expected value needs only these cell expectations."""
    lam, var = weights(np.asarray(x, float), np.asarray(y, float), m, ok=False)
    mu, sd = lam.T @ ns.scores, np.sqrt(var)
    t, wq = np.polynomial.hermite_e.hermegauss(C.N_QUAD)          # nodes/weights for N(0, 1)
    phi = ns.back(mu[:, None] + sd[:, None] * t[None, :])
    return (C.ooip_cells(phi) @ (wq / wq.sum())).reshape(C.NY, C.NX)

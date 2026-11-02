"""Sequential Gaussian simulation (SGS), GSLIB sgsim algorithm, compiled with numba.

Per realization: well normal scores are assigned to their grid nodes; every other node is visited
on a random path; at each node the nearest NODMAX informed nodes (wells and previously simulated
nodes) are found with a search template ordered by anisotropic distance; simple kriging (mean 0,
sill 1) gives the local mean and variance; a value is drawn from that Gaussian. Realizations are
back-transformed with the declustered normal-score table (wells.NormalScore).

GeostatsPy's sgsim implements the same algorithm in interpreted Python (~1 minute per realization
here); this version takes well under a second, which makes 100 realizations per variogram model
practical.
"""
import numpy as np
from numba import njit

from . import config as C

NODMAX = 24


@njit(cache=True)
def _cov(dx, dy, it, nug, azi, hmaj, hmin):
    if dx == 0.0 and dy == 0.0:
        return 1.0
    a = np.radians(azi)
    dmaj = dx * np.sin(a) + dy * np.cos(a)
    dmin = dx * np.cos(a) - dy * np.sin(a)
    r = np.sqrt((dmaj / hmaj) ** 2 + (dmin / hmin) ** 2)
    if it == 1:
        s = 1.5 * r - 0.5 * r ** 3 if r < 1.0 else 1.0
    elif it == 2:
        s = 1.0 - np.exp(-3.0 * r)
    else:
        s = 1.0 - np.exp(-3.0 * r * r)
    return (1.0 - nug) * (1.0 - s)


def _template(m):
    """Offsets (row, col) inside the search ellipse (the variogram ranges), nearest first."""
    rmax = int(min(max(C.NX, C.NY) - 1, np.ceil(max(m['hmaj'], 2 * C.XSIZ) / min(C.XSIZ, C.YSIZ))))
    dr, dc = np.mgrid[-rmax:rmax + 1, -rmax:rmax + 1]
    dr, dc = dr.ravel(), dc.ravel()
    a = np.radians(m['azi'])
    dx, dy = dc * C.XSIZ, -dr * C.YSIZ
    dmaj, dmin = dx * np.sin(a) + dy * np.cos(a), dx * np.cos(a) - dy * np.sin(a)
    hmaj, hmin = max(m['hmaj'], 2 * C.XSIZ), max(m['hmin'], 2 * C.XSIZ)
    r = np.sqrt((dmaj / hmaj) ** 2 + (dmin / hmin) ** 2)
    keep = (r <= 1.0) & ((dr != 0) | (dc != 0))
    o = np.lexsort((np.hypot(dx, dy)[keep], r[keep]))
    return dr[keep][o].astype(np.int64), dc[keep][o].astype(np.int64)


@njit(cache=True)
def _simulate(ny, nx, xsiz, ysiz, data_node, data_val, path, z, tr, tc, nodmax, it, nug, azi, hmaj, hmin):
    sim = np.zeros(ny * nx)
    done = np.zeros(ny * nx, dtype=np.bool_)
    for k in range(len(data_node)):
        sim[data_node[k]] = data_val[k]
        done[data_node[k]] = True
    nr = np.empty(nodmax, dtype=np.int64)
    nc = np.empty(nodmax, dtype=np.int64)
    for p in range(len(path)):
        node = path[p]
        if done[node]:
            continue
        r0, c0 = node // nx, node % nx
        n = 0
        for t in range(len(tr)):                          # nearest informed nodes, template order
            r, c = r0 + tr[t], c0 + tc[t]
            if r >= 0 and r < ny and c >= 0 and c < nx and done[r * nx + c]:
                nr[n], nc[n] = r, c
                n += 1
                if n == nodmax:
                    break
        mean, var = 0.0, 1.0
        if n > 0:
            K = np.empty((n, n)); kv = np.empty(n); v = np.empty(n)
            for i in range(n):
                v[i] = sim[nr[i] * nx + nc[i]]
                kv[i] = _cov((nc[i] - c0) * xsiz, -(nr[i] - r0) * ysiz, it, nug, azi, hmaj, hmin)
                for j in range(n):
                    K[i, j] = _cov((nc[i] - nc[j]) * xsiz, -(nr[i] - nr[j]) * ysiz, it, nug, azi, hmaj, hmin)
            lam = np.linalg.solve(K, kv)
            mean = 0.0
            var = 1.0
            for i in range(n):
                mean += lam[i] * v[i]
                var -= lam[i] * kv[i]
            var = max(var, 0.0)
        sim[node] = mean + np.sqrt(var) * z[p]
        done[node] = True
    return sim


def simulate_scores(x, y, scores, m, n_sim, rng, nodmax=NODMAX):
    """n_sim realizations of the normal-score field (n_sim, NY, NX), conditioned to the wells."""
    rows = np.rint((C.YMAX - np.asarray(y)) / C.YSIZ - 0.5).astype(np.int64)
    cols = np.rint((np.asarray(x) - C.XMIN) / C.XSIZ - 0.5).astype(np.int64)
    data_node = rows * C.NX + cols
    tr, tc = _template(m)
    out = np.empty((n_sim, C.NY, C.NX))
    for s in range(n_sim):
        path = rng.permutation(C.NX * C.NY).astype(np.int64)
        z = rng.standard_normal(C.NX * C.NY)
        out[s] = _simulate(C.NY, C.NX, C.XSIZ, C.YSIZ, data_node, np.asarray(scores, float), path, z, tr, tc,
                           nodmax, int(m['it']), float(m['nug']), float(m['azi']), float(m['hmaj']),
                           float(m['hmin'])).reshape(C.NY, C.NX)
    return out


def simulate(x, y, ns, m, n_sim, rng, nodmax=NODMAX):
    """Porosity realizations: SGS of the declustered normal scores, back-transformed.
    ns: wells.NormalScore of the well data; m: variogram model (sill 1) used for the normal scores."""
    return ns.back(simulate_scores(x, y, ns.scores, m, n_sim, rng, nodmax))

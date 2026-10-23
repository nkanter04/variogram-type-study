"""Experimental continuity measures from wells and the exhaustive variogram of a truth model.

Measures (GSLIB gamv numbering): 1 semivariogram, 4 correlogram, 5 general relative,
6 pairwise relative, 7 semivariogram of logarithms, 8 rodogram (GSLIB 2 / textbook definition),
plus the cumulative semivariogram (Sen 1989) and the semivariogram of normal scores.

gamv conventions: every pair once; direction cone +/- ATOL and a bandwidth; head = the point in
the +azimuth sense. Lags are spaced at the median nearest-neighbour well distance, so the first
lags always have pairs whatever the well count.
"""
import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from . import config as C
from .models import gamma

RAW = ['semivariogram', 'correlogram', 'general_relative', 'pairwise_relative', 'log', 'rodogram', 'normal_score']


def lag_settings(x, y):
    d, _ = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)
    xlag = max(20.0, 5.0 * round(np.median(d[:, 1]) / 5.0))
    return dict(xlag=xlag, xltol=xlag / 2.0, nlag=int(C.MAX_LAG // xlag), bandwh=C.BAND_FACTOR * xlag)


def _pairs(x, y):
    i, j = np.triu_indices(len(x), k=1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    return i, j, dx, dy, np.hypot(dx, dy)


def _lag_stats(t, hd, k, nb):
    """Per-lag sums for every raw measure (vectorized with bincount)."""
    s = lambda a: np.bincount(k, weights=a, minlength=nb)
    n = np.bincount(k, minlength=nb).astype(float)
    d = hd - t
    return n, dict(d2=s(d ** 2), dl2=s((np.log(hd) - np.log(t)) ** 2), pw=s((d / (0.5 * (hd + t))) ** 2),
                   rod=s(np.sqrt(np.abs(d))), t=s(t), h=s(hd), tt=s(t * t), hh=s(hd * hd), th=s(t * hd))


def _measures(n, S, ns):
    """Raw values per lag. ns = per-lag semivariance of normal scores (computed separately)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        mt, mh = S['t'] / n, S['h'] / n
        st = np.sqrt(np.maximum(S['tt'] / n - mt ** 2, 0)); sh = np.sqrt(np.maximum(S['hh'] / n - mh ** 2, 0))
        return {'semivariogram': 0.5 * S['d2'] / n,
                'correlogram': (S['th'] / n - mt * mh) / (st * sh),
                'general_relative': (S['d2'] / n) / (0.5 * (mt + mh)) ** 2,
                'pairwise_relative': 0.5 * S['pw'] / n,
                'log': 0.5 * S['dl2'] / n,
                'rodogram': 0.5 * S['rod'] / n,
                'normal_score': ns}


def _standardize(val, vt, sill):
    if vt == 'correlogram':
        return 1.0 - val
    if vt == 'rodogram':                       # rodogram ~ gamma^(1/4) under Gaussian increments
        return (val / sill) ** 4
    return val / sill


def experimental(x, y, v, scores, lags=None, azimuths=C.AZIMUTHS):
    """All measures along every candidate azimuth -> tidy DataFrame
    (vtype, azi, lag, h, npairs, value, std_value), cumulative included."""
    x, y, v, scores = (np.asarray(a, float) for a in (x, y, v, scores))
    lags = lags or lag_settings(x, y)
    i, j, dx, dy, h = _pairs(x, y)
    nb = lags['nlag'] + 1
    # sills: the same measure over all pairs regardless of separation
    _, Sall = _lag_stats(v[i], v[j], np.zeros(len(i), int), 1)
    sills = {k: float(val[0]) for k, val in _measures(np.array([len(i)], float), Sall,
                                                       np.array([0.5 * np.mean((scores[j] - scores[i]) ** 2)])).items()}
    out = []
    for azi in azimuths:
        ux, uy = np.sin(np.radians(azi)), np.cos(np.radians(azi))
        along = dx * ux + dy * uy
        ok = (np.abs(along) >= np.cos(np.radians(C.ATOL)) * h) & (np.abs(dx * uy - dy * ux) <= lags['bandwh'])
        k = np.floor(h / lags['xlag'] + 0.5).astype(int)
        ok &= (k <= lags['nlag']) & (np.abs(h - k * lags['xlag']) <= lags['xltol'])
        fwd = along[ok] >= 0
        ii, jj = np.where(fwd, i[ok], j[ok]), np.where(fwd, j[ok], i[ok])     # tail, head
        kk = k[ok]
        n, S = _lag_stats(v[ii], v[jj], kk, nb)
        ns = 0.5 * np.bincount(kk, weights=(scores[jj] - scores[ii]) ** 2, minlength=nb) / np.where(n > 0, n, np.nan)
        hbar = np.bincount(kk, weights=h[ok], minlength=nb) / np.where(n > 0, n, np.nan)
        for vt, val in _measures(n, S, ns).items():
            out.append(pd.DataFrame(dict(vtype=vt, azi=azi, lag=np.arange(nb), h=hbar, npairs=n,
                                         value=val, std_value=_standardize(val, vt, sills[vt]))))
    e = pd.concat(out, ignore_index=True)
    e = e[(e.npairs > 0)]
    return pd.concat([e, _cumulative(e)], ignore_index=True)


def _cumulative(e):
    """Cumulative semivariogram Gamma(h_k) = sum_{j<=k} gamma(h_j) dh_j over usable lags;
    the fitted quantity is its running mean Gamma(h_k) / h_k."""
    parts = []
    for azi, s in e[(e.vtype == 'semivariogram') & (e.npairs >= C.MIN_PAIRS)].groupby('azi'):
        s = s.dropna(subset=['std_value']).sort_values('h')
        dh = np.diff(np.r_[0.0, s.h.values])
        g = np.cumsum(s.std_value.values * dh)
        parts.append(s.assign(vtype='cumulative', value=g, std_value=g / s.h.values))
    return pd.concat(parts, ignore_index=True) if parts else e.iloc[:0]


# ------------------------------------------------------------------ exhaustive truth variogram
def variogram_map(z):
    """Exhaustive semivariogram for every whole-cell offset (FFT, Marcotte 1996), zero offset at
    the centre; axis 0 = row offset (+ = south), axis 1 = column offset (+ = east)."""
    s = (2 * z.shape[0], 2 * z.shape[1])
    one = np.ones_like(z)
    F = lambda a: sfft.rfft2(a, s=s)
    cc = lambda a, b: sfft.irfft2(np.conj(F(a)) * F(b), s=s)
    g = 0.5 * (cc(one, z ** 2) + cc(z ** 2, one) - 2.0 * cc(z, z)) / np.maximum(cc(one, one), 1)
    return np.fft.fftshift(g)[1:, 1:]


# offsets within SCORE_HSHORT, one of each +/- pair: used to score a model against the truth map
_r = int(C.SCORE_HSHORT // C.XSIZ)
_dr, _dc = np.mgrid[-_r:_r + 1, -_r:_r + 1]
_keep = (np.hypot(_dr * C.YSIZ, _dc * C.XSIZ) <= C.SCORE_HSHORT) & ((_dr > 0) | ((_dr == 0) & (_dc > 0)))
SHORT_DR, SHORT_DC = _dr[_keep], _dc[_keep]


class Truth:
    """Exhaustive variogram of one truth model, standardized by its variance."""

    def __init__(self, phi):
        self.map = variogram_map(phi) / phi.var()
        self.short = self.map[C.NY - 1 + SHORT_DR, C.NX - 1 + SHORT_DC]

    def short_rmse(self, m):
        """RMSE between model m and the truth over every lag vector up to SCORE_HSHORT (all directions)."""
        g = gamma(SHORT_DC * C.XSIZ, -SHORT_DR * C.YSIZ, m)
        return float(np.sqrt(np.mean((g - self.short) ** 2)))

    def curves(self, h=np.arange(C.XSIZ, C.MAX_LAG + 1e-9, C.XSIZ), azimuths=C.AZIMUTHS):
        """Truth variogram along each candidate azimuth (bilinear), in the experimental-table format."""
        rows = []
        for azi in azimuths:
            a = np.radians(azi)
            r = (C.NY - 1) - h * np.cos(a) / C.YSIZ
            c = (C.NX - 1) + h * np.sin(a) / C.XSIZ
            g = map_coordinates(self.map, [r, c], order=1, mode='nearest')
            rows.append(pd.DataFrame(dict(vtype='truth', azi=azi, h=h, npairs=float(C.MIN_PAIRS), std_value=g)))
        return pd.concat(rows, ignore_index=True)

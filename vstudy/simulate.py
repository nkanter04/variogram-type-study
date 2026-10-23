"""Gaussian random fields by FFT circulant embedding (Dietrich & Newsam 1997) and truth models.

Exact for a stationary covariance on a regular grid and ~10 ms per 100 x 100 field, which is what
makes 30 truth models per scenario and 50 conditional simulations per dataset affordable.
"""
import numpy as np
from scipy import fft as sfft
from scipy.stats import norm

from . import config as C
from .models import covariance

_CACHE = {}


def _embedding(m):
    key = (int(m['it']), float(m['azi']), float(m['hmaj']), float(m['hmin']))
    if key not in _CACHE:
        reach = max(m['hmaj'], m['hmin']) * (1.0 if int(m['it']) == 1 else 2.5)
        ny = sfft.next_fast_len(max(2 * C.NY, C.NY + int(np.ceil(reach / C.YSIZ)) + 1))
        nx = sfft.next_fast_len(max(2 * C.NX, C.NX + int(np.ceil(reach / C.XSIZ)) + 1))
        kr = np.arange(ny); kr = np.where(kr <= ny // 2, kr, kr - ny)
        kc = np.arange(nx); kc = np.where(kc <= nx // 2, kc, kc - nx)
        dc, dr = np.meshgrid(kc, kr)
        c = covariance(dc * C.XSIZ, -dr * C.YSIZ, dict(m, nug=0.0))      # +row offset = south
        lam = sfft.fft2(c).real
        _CACHE[key] = np.sqrt(np.clip(lam, 0.0, None) / lam.size)
    return _CACHE[key]


def gaussian_fields(m, n, rng):
    """n unconditional N(0,1) fields of shape (NY, NX) with standardized variogram model m."""
    sq = _embedding(m)
    out = []
    while len(out) < n:
        z = sfft.fft2(sq * (rng.standard_normal(sq.shape) + 1j * rng.standard_normal(sq.shape)))
        out += [z.real[:C.NY, :C.NX], z.imag[:C.NY, :C.NX]]            # two independent fields per FFT
    f = np.array(out[:n])
    nug = float(m['nug'])
    return np.sqrt(1.0 - nug) * f + np.sqrt(nug) * rng.standard_normal(f.shape)


def truth_rng(stage, scenario, real):
    """Reproducible generator for one truth model; pilot and main stages never share seeds."""
    base = C.SEED_PILOT if stage == 'pilot' else C.SEED_MAIN
    return np.random.default_rng([base, C.SCEN_NAMES.index(scenario), real])


def make_truth(scenario, rng):
    """Porosity grid (NY, NX) and facies grid (or None) for one scenario."""
    s = C.SCENARIOS[scenario]

    def field(v):
        y = gaussian_fields(v, 1, rng)[0]
        return (y - y.mean()) / y.std()

    y = field(s['vario'])
    facies = None
    if s['transform'] == 'gaussian':
        phi = s['mean'] + s['std'] * y
    elif s['transform'] == 'lognormal':
        phi = s['median'] * np.exp(s['log_std'] * y)
    elif s['transform'] == 'facies':
        sand = field(s['facies_vario']) > norm.ppf(1.0 - s['sand_prop'])
        phi = np.where(sand, s['sand'][0] + s['sand'][1] * y, s['shale'][0] + s['shale'][1] * field(s['vario']))
        facies = sand.astype(int)
    else:
        raise ValueError(s['transform'])
    return np.clip(phi, C.PHI_LO, C.PHI_HI), facies

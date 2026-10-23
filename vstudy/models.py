"""Variogram / covariance models (standardized, sill 1) with 2-D geometric anisotropy.

A model is a dict with keys it, nug, azi, hmaj, hmin (GSLIB conventions).
dx is east, dy is north, both in metres.
"""
import numpy as np


def unit_structure(it, r):
    """Standardized structure (0 at r=0, 1 at/after the practical range) for normalized lag r = h/a."""
    it = int(it)
    if it == 1:
        return np.where(r < 1.0, 1.5 * r - 0.5 * r ** 3, 1.0)
    if it == 2:
        return 1.0 - np.exp(-3.0 * r)
    if it == 3:
        return 1.0 - np.exp(-3.0 * r ** 2)
    raise ValueError(f'unknown structure it={it}')


def aniso_r(dx, dy, m):
    """Normalized anisotropic distance (1.0 = at the range)."""
    az = np.radians(m['azi'])
    d_maj = dx * np.sin(az) + dy * np.cos(az)
    d_min = dx * np.cos(az) - dy * np.sin(az)
    return np.sqrt((d_maj / m['hmaj']) ** 2 + (d_min / m['hmin']) ** 2)


def gamma(dx, dy, m):
    """Standardized variogram value for separation (dx, dy); 0 at the origin."""
    dx, dy = np.asarray(dx, float), np.asarray(dy, float)
    g = m['nug'] + (1.0 - m['nug']) * unit_structure(m['it'], aniso_r(dx, dy, m))
    return np.where(np.hypot(dx, dy) < 1e-9, 0.0, g)


def covariance(dx, dy, m, sill=1.0):
    """C(h) = sill * (1 - gamma(h)); C(0) = sill (includes the nugget)."""
    return sill * (1.0 - gamma(dx, dy, m))


def gamma_along(h, azi_dir, m):
    """Standardized model evaluated at distances h along azimuth azi_dir."""
    a = np.radians(azi_dir)
    h = np.asarray(h, float)
    return gamma(h * np.sin(a), h * np.cos(a), m)


def range_along(azi_dir, m):
    d = np.radians(azi_dir - m['azi'])
    return 1.0 / np.sqrt((np.cos(d) / m['hmaj']) ** 2 + (np.sin(d) / m['hmin']) ** 2)


def as_model(row):
    """Pull a model dict out of a pandas row / dict."""
    return dict(it=int(row['it']), nug=float(row['nug']), azi=float(row['azi']),
                hmaj=float(row['hmaj']), hmin=float(row['hmin']))

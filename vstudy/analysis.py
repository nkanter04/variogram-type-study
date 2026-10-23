"""Summary tables used by the notebooks. All comparisons are paired by truth model (the unit of
replication) with 95% confidence intervals; see stats.py."""
import numpy as np
import pandas as pd
from scipy import stats as st

from . import config as C
from .stats import paired_table, TRUTH_KEY

KEY = ['scenario', 'realization', 'n_wells', 'design']
ALT = C.VTYPES[1:] + ['loocv']                       # alternatives compared with the semivariogram


def by(df, metric, rows, cols='method', columns=None):
    t = df.pivot_table(index=rows, columns=cols, values=metric, aggfunc='mean')
    return t[columns] if columns is not None else t


def paired_by_wells(df, metric, methods=ALT, col='method'):
    """{'25 wells': table, ...}: each method minus the semivariogram, CI clustered by truth, Holm."""
    return {f'{n} wells': paired_table(df[df.n_wells == n], metric, methods, col=col)
            for n in sorted(df.n_wells.unique())}


def truth_level_diff(df, metric, a, b):
    """Mean of (rows matching a) - (rows matching b), averaged within each truth model first.
    a, b: dicts of column -> value. Returns (mean, ci_lo, ci_hi, n_truths)."""
    sel = lambda f: df.loc[np.logical_and.reduce([df[k] == v for k, v in f.items()])].groupby(TRUTH_KEY)[metric].mean()
    d = (sel(a) - sel(b)).dropna()
    half = st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - half, d.mean() + half, len(d)


def value_of_information(dec, n_ref=50):
    """$MM per 3-well program gained by each lever (lower regret = value), with 95% CIs."""
    best = dec[(dec.n_wells == n_ref) & dec.method.isin(C.VTYPES)].groupby('method').regret_MM.mean().idxmin()
    levers = [
        (f'Any map (inverse distance) vs random sites, {n_ref} wells', dict(method='random', n_wells=n_ref), dict(method='idw', n_wells=n_ref)),
        ('25 -> 50 appraisal wells (semivariogram)', dict(method='semivariogram', n_wells=25), dict(method='semivariogram', n_wells=50)),
        ('50 -> 100 appraisal wells (semivariogram)', dict(method='semivariogram', n_wells=50), dict(method='semivariogram', n_wells=100)),
        (f'True variogram vs fitted semivariogram, {n_ref} wells', dict(method='semivariogram', n_wells=n_ref), dict(method='truth_variogram', n_wells=n_ref)),
        (f'Best type in hindsight ({C.LABEL[best]}) vs semivariogram', dict(method='semivariogram', n_wells=n_ref), dict(method=best, n_wells=n_ref)),
        (f'Cross-validation pick vs semivariogram, {n_ref} wells', dict(method='semivariogram', n_wells=n_ref), dict(method='loocv', n_wells=n_ref)),
        (f'Kriging (semivariogram) vs inverse distance, {n_ref} wells', dict(method='idw', n_wells=n_ref), dict(method='semivariogram', n_wells=n_ref)),
    ]
    rows = []
    for name, a, b in levers:
        m, lo, hi, g = truth_level_diff(dec, 'regret_MM', a, b)
        rows.append(dict(lever=name, value_MM=m, ci_lo=lo, ci_hi=hi, n_truths=g))
    return pd.DataFrame(rows).set_index('lever')


def same_sites(dec, methods=C.VTYPES):
    """% of datasets in which two methods choose exactly the same 3 well sites."""
    piv = dec[dec.method.isin(methods)].pivot_table(index=KEY, columns='method', values='sites', aggfunc='first')
    return pd.DataFrame({a: {b: 100 * (piv[a] == piv[b]).mean() for b in methods} for a in methods}).loc[methods, methods]


def trimmed_types(pilot_dec, threshold=C.TRIM_SAME_SITES):
    """Pre-stated rule: a type is merged into an earlier type in VTYPES if they choose identical
    sites in >= threshold % of pilot datasets. Returns (kept types, {merged: kept_as})."""
    s = same_sites(pilot_dec)
    kept, merged = [], {}
    for vt in C.VTYPES:
        twin = next((k for k in kept if s.loc[vt, k] >= threshold), None)
        if twin:
            merged[vt] = twin
        else:
            kept.append(vt)
    return kept, merged

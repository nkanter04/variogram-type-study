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


# ------------------------------------------------------------------ estimation by simulation (SGS)
RULES = {'kriged map': '', 'SGS expected value': '@sgs_ev', 'SGS risk-averse (P10)': '@sgs_p10'}


def rule_frame(dec, suffix):
    """Decision rows of one rule, with the rule suffix stripped from the method name."""
    if suffix:
        d = dec[dec.method.str.endswith(suffix)].copy()
        d['method'] = d.method.str[:-len(suffix)]
        return d
    return dec[~dec.method.str.contains('@')]


def type_separation(dec):
    """How differently the 8 variogram types decide under each rule, per well count:
    spread of mean regret across types, % of datasets where a type picks the same sites as the
    semivariogram (average over the 7 alternatives), and how many types differ significantly from the
    semivariogram (Holm)."""
    rows = []
    for rule, suf in RULES.items():
        d = rule_frame(dec, suf)
        for n in sorted(d.n_wells.unique()):
            dn = d[d.n_wells == n]
            mean = dn[dn.method.isin(C.VTYPES)].groupby('method').regret_MM.mean()
            piv = dn[dn.method.isin(C.VTYPES)].pivot_table(index=KEY, columns='method', values='sites', aggfunc='first')
            same = np.mean([(piv[t] == piv['semivariogram']).mean() for t in C.VTYPES[1:]]) * 100
            pt = paired_table(dn, 'regret_MM', C.VTYPES[1:])
            rows.append({'rule': rule, 'n_wells': n, 'spread of mean regret across types ($MM)': mean.max() - mean.min(),
                         'same sites as semivariogram (%)': same,
                         'types significantly better': int(((pt.p_holm < 0.05) & (pt.mean_diff < 0)).sum()),
                         'types significantly worse': int(((pt.p_holm < 0.05) & (pt.mean_diff > 0)).sum())})
    return pd.DataFrame(rows).set_index(['rule', 'n_wells'])


def rule_vs_kriging(dec, methods=('semivariogram', 'normal_score', 'loocv', 'truth_variogram')):
    """Regret of each SGS rule minus the kriged-map decision with the same variogram model ($MM),
    averaged within truth models, with 95% CI."""
    rows = []
    for rule, suf in list(RULES.items())[1:]:
        for m in methods:
            for n in sorted(dec.n_wells.unique()):
                mean, lo, hi, g = truth_level_diff(dec, 'regret_MM', dict(method=m + suf, n_wells=n), dict(method=m, n_wells=n))
                rows.append(dict(rule=rule, method=m, n_wells=n, diff_MM=mean, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows).set_index(['rule', 'method', 'n_wells'])


# ------------------------------------------------------------------ areas of interest (fixed before modelling)
AOI_ORDER = list(C.AOIS)
AOI_RULES = {'kriged map': 'kriged map', 'SGS expected value': 'sgs_ev', 'SGS risk-averse (P10)': 'sgs_p10'}


def aoi_accuracy(aoi, source='map', methods=C.VTYPES + ['loocv', 'truth_variogram', 'idw', 'naive']):
    """Mean absolute OOIP error (%) in each AOI, by estimation method."""
    a = aoi[(aoi.source == source) & aoi.method.isin(methods)]
    t = a.pivot_table(index='aoi', columns='method', values='abs_err_pct', aggfunc='mean')
    return t.reindex(index=AOI_ORDER, columns=[m for m in methods if m in t.columns])


def aoi_type_spread(aoi, source='map'):
    """How far apart the 8 variogram types put the AOI OOIP: (max - min) / true OOIP, %, averaged over
    datasets, by AOI and well count."""
    a = aoi[(aoi.source == source) & aoi.method.isin(C.VTYPES)]
    g = a.groupby(KEY + ['aoi'])
    sp = (100 * (g.est_ooip.max() - g.est_ooip.min()) / g.true_ooip.first()).rename('spread').reset_index()
    return sp.pivot_table(index='aoi', columns='n_wells', values='spread', aggfunc='mean').reindex(AOI_ORDER)


def aoi_decision_summary(dec, methods=C.VTYPES + ['loocv', 'truth_variogram', 'idw']):
    """Per rule and method: mean regret ($MM), % choosing the truly best local AOI, % same choice as the semivariogram."""
    rows = []
    for label, rule in AOI_RULES.items():
        d = dec[dec.rule == rule]
        semi = d[d.method == 'semivariogram'].set_index(KEY).chosen
        for m in methods:
            dm = d[d.method == m].set_index(KEY)
            if not len(dm):
                continue
            rows.append(dict(rule=label, method=m, regret_MM=dm.regret_MM.mean(), pct_best=100 * (dm.chosen == dm.best).mean(),
                             pct_same_as_semi=100 * (dm.chosen == semi.reindex(dm.index)).mean()))
    return pd.DataFrame(rows).set_index(['rule', 'method'])


def aoi_paired(dec, methods=C.VTYPES[1:] + ['loocv', 'truth_variogram', 'idw']):
    """{rule: paired table}: AOI-decision regret of each method minus the semivariogram (clustered, Holm)."""
    return {label: paired_table(dec[(dec.rule == rule) & dec.method.isin(methods + ['semivariogram'])], 'regret_MM',
                                [m for m in methods if (dec[dec.rule == rule].method == m).any()])
            for label, rule in AOI_RULES.items()}

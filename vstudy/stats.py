"""Paired comparisons that respect the study's structure.

The original t-tests treat every (scenario, realization, design) dataset as independent. The
designs (and well counts) share one truth model, so they are correlated; the honest unit of
replication is the truth model. Here the paired difference is averaged within each truth model
first (cluster mean), and the t-test is run across truth models. Holm's method controls the
family-wise error over the alternatives compared in one table.
"""
import numpy as np
import pandas as pd
from scipy import stats as st

TRUTH_KEY = ['scenario', 'realization']


def dataset_key(df):
    return [c for c in ('scenario', 'realization', 'n_wells', 'design') if c in df.columns]


def paired(df, metric, method, base='semivariogram', col='method'):
    """Difference method - base on `metric`, matched by dataset."""
    key = dataset_key(df)
    a = df[df[col] == method].set_index(key)[metric]
    b = df[df[col] == base].set_index(key)[metric]
    d = (a - b).dropna()
    n = len(d)
    out = dict(method=method, n=n, mean_diff=d.mean(), pct_better=100 * (d < 0).mean(),
               pct_same=100 * (np.abs(d) < 1e-12).mean())
    sd = d.std(ddof=1)
    out['t_naive'] = d.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    g = d.groupby(level=TRUTH_KEY).mean()                       # one value per truth model
    G, sg = len(g), g.std(ddof=1)
    out['n_truths'] = G
    out['t_cluster'] = g.mean() / (sg / np.sqrt(G)) if G > 1 and sg > 0 else np.nan
    out['p_cluster'] = 2 * st.t.sf(abs(out['t_cluster']), G - 1) if np.isfinite(out['t_cluster']) else np.nan
    half = st.t.ppf(0.975, G - 1) * sg / np.sqrt(G) if G > 1 else np.nan
    out['ci_lo'], out['ci_hi'] = g.mean() - half, g.mean() + half
    return out


def holm(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    adj = np.full_like(p, np.nan)
    idx = np.argsort(np.where(ok, p, np.inf))
    m, running = ok.sum(), 0.0
    for rank, i in enumerate(idx[:m]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def paired_table(df, metric, methods, base='semivariogram', col='method'):
    t = pd.DataFrame([paired(df, metric, m, base, col) for m in methods]).set_index('method')
    t['p_holm'] = holm(t.p_cluster.values)
    return t


def cluster_bootstrap(df, fn, n_boot=2000, seed=0):
    """Percentile CI of a statistic fn(df) by resampling whole truth models."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(TRUTH_KEY)]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(fn(pd.concat([groups[i] for i in pick], ignore_index=True)))
    return np.percentile(vals, [2.5, 97.5])

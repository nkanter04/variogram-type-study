"""Apply the pre-registered test (hypotheses.json) and write RESULTS.md.

    python confirmatory/analyze.py            # full results
    python confirmatory/analyze.py --smoke    # smoke-test results (not a finding)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))


def holm(p):
    p = np.asarray(p, float)
    adj, running, m = np.empty_like(p), 0.0, len(p)
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def per_truth_diff(df, scenarios, n_wells, method):
    s = df[df.scenario.isin(scenarios) & (df.n_wells == n_wells)]
    g = lambda m: s[s.method == m].groupby(['scenario', 'realization']).regret_MM.mean()
    return (g(method) - g('semivariogram')).dropna()


def main(smoke):
    hyp = json.load(open(os.path.join(HERE, 'hypotheses.json')))
    folder = os.path.join(HERE, 'smoke_results' if smoke else 'results')
    df = pd.read_csv(os.path.join(folder, 'regret.csv'))
    info = json.load(open(os.path.join(folder, 'run_info.json')))
    eps = hyp['smallest_effect_of_interest_MM']
    rows = []
    for h in hyp['hypotheses']:
        d = per_truth_diff(df, h['scenarios'], h['n_wells'], h['method'])
        n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se
        q95, q90 = st.t.ppf(0.975, n - 1), st.t.ppf(0.95, n - 1)
        rows.append(dict(id=h['id'], claim=h['claim'], n_truths=n, mean_diff_MM=m,
                         ci95_lo=m - q95 * se, ci95_hi=m + q95 * se, ci90_lo=m - q90 * se, ci90_hi=m + q90 * se,
                         t=t, p=2 * st.t.sf(abs(t), n - 1), pct_truths_better=100 * (d < 0).mean()))
    r = pd.DataFrame(rows).set_index('id')
    r['p_holm'] = holm(r.p.values)
    r['verdict'] = np.select(
        [(r.p_holm < 0.05) & (r.mean_diff_MM < 0), (r.p_holm < 0.05) & (r.mean_diff_MM > 0),
         (r.ci90_lo > -eps) & (r.ci90_hi < eps)],
        ['CONFIRMED', 'REVERSED', 'EQUIVALENT'], 'INCONCLUSIVE')
    r.to_csv(os.path.join(folder, 'hypothesis_tests.csv'))

    desc = df.pivot_table(index=['scenario', 'n_wells'], columns='method', values='regret_MM', aggfunc='mean')
    order = [c for c in ['semivariogram', 'normal_score', 'pairwise_relative', 'rodogram', 'general_relative', 'loocv',
                         'indicator_continuous', 'indicator_categorical', 'truth_variogram', 'idw'] if c in desc.columns]
    desc = desc[order]

    def md(t, fmt):
        cols = [t.index.name or ''] + [str(c) for c in t.columns]
        out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
        for i, row in t.iterrows():
            out.append('| ' + ' | '.join([str(i)] + [fmt(v) for v in row.values]) + ' |')
        return '\n'.join(out)

    f = lambda v: f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
    tab = r[['claim', 'n_truths', 'mean_diff_MM', 'ci95_lo', 'ci95_hi', 'p', 'p_holm', 'pct_truths_better', 'verdict']]
    desc.index = [f'{s}, {n} wells' for s, n in desc.index]
    desc.index.name = 'scenario'
    text = [f"# Confirmatory results{' (SMOKE TEST - not a finding)' if smoke else ''}", '',
            f"Pre-registered in `hypotheses.json`. Seed {info['seed']}, {info['n_truths_per_scenario']} truth models per scenario, "
            f"run time {info['runtime_s']} s. Differences are method minus semivariogram regret in $MM per 3-well program "
            f"(negative = method better), averaged within each truth model. Holm correction over H1-H7; "
            f"equivalence margin +/- {eps} $MM (90% CI).", '',
            '## Pre-registered hypotheses', '', md(tab, f), '',
            '## Mean regret by scenario ($MM per program, descriptive)', '', md(desc, f), '']
    open(os.path.join(folder, 'RESULTS.md'), 'w', encoding='utf-8').write('\n'.join(text))
    print(tab.to_string(float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    main(ap.parse_args().smoke)

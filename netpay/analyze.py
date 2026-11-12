"""Apply the pre-registered tests (hypotheses.json) and write RESULTS.md.

    python netpay/analyze.py            # pre-registered results
    python netpay/analyze.py --pilot    # exploratory pilot (not a finding)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH = ['scenario', 'realization']


def holm(p):
    p = np.asarray(p, float)
    adj, running, m = np.empty_like(p), 0.0, len(p)
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def load(folder):
    d = pd.read_csv(os.path.join(folder, 'decisions.csv'))
    d['abs_net_err_pct'] = 100 * (d.field_net - d.true_field_net).abs() / d.true_field_net
    return d


def per_truth(d, sel, metric, scenarios, n_wells):
    s = d[(d.method == sel['method']) & (d.rule == sel['rule']) & d.scenario.isin(scenarios) & d.n_wells.isin(n_wells)]
    return s.groupby(TRUTH)[metric].mean()


def test(d, h, default_scen, default_wells):
    scen, wells = h.get('scenarios', default_scen), h.get('n_wells', default_wells)
    diff = (per_truth(d, h['a'], h['metric'], scen, wells) - per_truth(d, h['b'], h['metric'], scen, wells)).dropna()
    n, m, se = len(diff), diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
    q95, q90 = st.t.ppf(0.975, n - 1), st.t.ppf(0.95, n - 1)
    return dict(id=h['id'], claim=h['claim'], metric=h['metric'], n_truths=n, mean_diff=m, sd=diff.std(ddof=1),
                ci95_lo=m - q95 * se, ci95_hi=m + q95 * se, ci90_lo=m - q90 * se, ci90_hi=m + q90 * se,
                p=2 * st.t.sf(abs(m / se), n - 1), pct_truths_better=100 * (diff < 0).mean())


def md(t, fmt=lambda v: f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)):
    cols = [t.index.name or ''] + [str(c) for c in t.columns]
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for i, row in t.iterrows():
        out.append('| ' + ' | '.join([str(i)] + [fmt(v) for v in row.values]) + ' |')
    return '\n'.join(out)


def main(pilot):
    hyp = json.load(open(os.path.join(HERE, 'hypotheses.json')))
    folder = os.path.join(HERE, 'pilot_results' if pilot else 'results')
    d = load(folder)
    info = json.load(open(os.path.join(folder, 'run_info.json')))
    scen, wells = sorted(d.scenario.unique()), hyp['primary_n_wells']
    r = pd.DataFrame([test(d, h, scen, wells) for h in hyp['hypotheses']]).set_index('id')
    r['p_holm'] = holm(r.p.values)
    eps = r.metric.map(hyp['equivalence_margin'])
    r['verdict'] = np.select([(r.p_holm < 0.05) & (r.mean_diff < 0), (r.p_holm < 0.05) & (r.mean_diff > 0),
                              (r.ci90_lo > -eps) & (r.ci90_hi < eps)], ['CONFIRMED', 'REVERSED', 'EQUIVALENT'], 'INCONCLUSIVE')
    r.to_csv(os.path.join(folder, 'hypothesis_tests.csv'))

    order = [m for m in hyp['method_order'] if m in d.method.unique()]
    reg = d.pivot_table(index='method', columns=['rule', 'n_wells'], values='regret_MM').reindex(order)
    net = d[d.rule.isin(['map_net', 'etype_net', 'sgs_net'])].pivot_table(index='method', columns=['rule', 'n_wells'], values='abs_net_err_pct').reindex(order)
    f = pd.read_csv(os.path.join(folder, 'fits.csv'))
    acc = f.pivot_table(index='method', columns='n_wells', values=['vario_rmse', 'azi_err']).reindex(order)
    for t in (reg, net, acc):
        t.columns = [', '.join(map(str, c)) for c in t.columns]

    tab = r[['claim', 'n_truths', 'mean_diff', 'ci95_lo', 'ci95_hi', 'p_holm', 'pct_truths_better', 'verdict']]
    text = [f"# Net-pay and maximum-likelihood results{' (EXPLORATORY PILOT - not a finding)' if pilot else ''}", '',
            f"Seed {info['seed']}, {info['n_truths_per_scenario']} truth models per scenario, run time {info['runtime_s']} s. "
            f"Differences are A minus B (negative = A better), averaged within each truth model over both well designs, "
            f"at {wells} wells unless the hypothesis says otherwise. Holm correction over all hypotheses. "
            f"Regret in $MM per 3-well program; net-OOIP error in % of the true field net OOIP.", '',
            '## Pre-registered hypotheses', '', md(tab), '',
            '## Mean regret ($MM per program) by rule and well count', '',
            '`map_total`: wells placed on the kriged map, all OOIP counts (the main study\'s decision). '
            '`map_net`: wells placed on the kriged map, net OOIP only. `etype_net`: wells placed on the expected net OOIP. '
            '`sgs_net`: the same from 50 SGS realizations.', '', md(reg), '',
            '## Field net OOIP: mean absolute error (%)', '', md(net), '',
            '## Variogram model accuracy: azimuth error (deg, anisotropic scenarios) and short-range RMSE vs truth', '', md(acc), '']
    open(os.path.join(folder, 'RESULTS.md'), 'w', encoding='utf-8').write('\n'.join(text))
    pd.set_option('display.width', 250)
    print(tab.drop(columns='claim').to_string(float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    main(ap.parse_args().pilot)

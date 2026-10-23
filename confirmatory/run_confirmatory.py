"""Generate the confirmatory data and decisions.

    python confirmatory/run_confirmatory.py            # full run: 200 truths per scenario, seed 31337
    python confirmatory/run_confirmatory.py --smoke    # 3 truths per scenario, seed 999, discarded output
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from vstudy import config as C, crossval, decisions as D, fitting, kriging, wells as W   # noqa: E402
from vstudy.variography import Truth, experimental                                       # noqa: E402
import scenarios as S                                                                     # noqa: E402

HYP = json.load(open(os.path.join(HERE, 'hypotheses.json')))


def run_truth(args):
    scenario, real, seed = args
    si = S.SCENARIOS.index(scenario)
    phi, facies = S.truth(scenario, np.random.default_rng([seed, si, real]))
    dv_true = D.drainage_value(phi)
    v_opt = D.program_value(D.place(dv_true, 'swap'), dv_true)
    ref = fitting.fit(Truth(phi).curves())
    rows = []
    for n in S.WELLS[scenario]:
        for di, design in enumerate(C.DESIGNS):
            rng = np.random.default_rng([seed, si, real, n, di, 1])
            w = W.sample(phi, design, n, rng)
            x, y = w['x'], w['y']
            v = S.recorded_values(scenario, w['v'], rng)
            wt = W.decluster(x, y, v, preferential=(design == 'preferential'))
            _, var_d = W.weighted_stats(v, wt)
            ns = W.NormalScore(v, wt)
            models = fitting.fit_all(experimental(x, y, v, ns.scores))
            choice, _ = crossval.pick(models, x, y, v)
            meth = {k: models[k] for k in ('semivariogram', 'normal_score', 'pairwise_relative', 'rodogram', 'general_relative')}
            meth['loocv'] = models[choice]
            meth['indicator_continuous'] = S.indicator_model(x, y, (v <= np.median(v)).astype(float))
            if facies is not None:
                meth['indicator_categorical'] = S.indicator_model(x, y, facies[w['row'], w['col']])
            meth['truth_variogram'] = {k: ref[k] for k in ('it', 'nug', 'azi', 'hmaj', 'hmin')}
            maps = {k: kriging.ordinary_kriging(x, y, v, m, var_d)[0] for k, m in meth.items()}
            maps['idw'] = kriging.idw(x, y, v)
            for k, est in maps.items():
                sites = D.place(D.drainage_value(est), 'swap')
                tv = D.program_value(sites, dv_true)
                rows.append(dict(scenario=scenario, realization=real, n_wells=n, design=design, method=k,
                                 opt_value=v_opt, true_value=tv, regret_pct=100 * (v_opt - tv) / v_opt,
                                 regret_MM=C.to_money(v_opt - tv), loocv_choice=choice if k == 'loocv' else None,
                                 rmse_phi=float(np.sqrt(np.mean((est - phi) ** 2)))))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    seed, n_real = (999, 3) if a.smoke else (HYP['seed'], HYP['n_truths_per_scenario'])
    out = os.path.join(HERE, 'smoke_results' if a.smoke else 'results')
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'
    jobs = [(s, r, seed) for s in S.SCENARIOS for r in range(n_real)]
    print(f'{len(jobs)} truth models, seed {seed}, {a.workers} processes', flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run_truth, jobs, chunksize=4), 1):
            rows += r
            if i % max(1, len(jobs) // 10) == 0:
                print(f'  {i}/{len(jobs)} ({time.time() - t0:.0f} s)', flush=True)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'regret.csv'), index=False)
    json.dump(dict(seed=seed, n_truths_per_scenario=n_real, runtime_s=round(time.time() - t0)),
              open(os.path.join(out, 'run_info.json'), 'w'), indent=2)
    print(f'saved {out} ({time.time() - t0:.0f} s)')

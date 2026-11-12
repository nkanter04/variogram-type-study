"""Generate data and decisions for the net-pay / maximum-likelihood study.

    python netpay/run_netpay.py --pilot     # exploratory: 10 truths per scenario, seed 555 -> pilot_results/
    python netpay/run_netpay.py             # pre-registered run: design read from hypotheses.json -> results/
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
from vstudy import config as C, decisions as D, fitting, kriging, sgs, wells as W     # noqa: E402
from vstudy.simulate import make_truth                                                  # noqa: E402
from vstudy.variography import Truth, experimental, lag_settings                        # noqa: E402
import core                                                                              # noqa: E402

N_SGS = 50
KEYS = ('it', 'nug', 'azi', 'hmaj', 'hmin')


def _decide(score_map, dv_true, v_opt):
    """Place 3 wells on a per-cell value map; return regret ($MM) against the true drained value."""
    s = D.place(D.drainage_from_ooip(score_map), 'swap')
    return C.to_money(v_opt - D.program_value(s, dv_true))


def run_truth(args):
    scenario, real, seed = args
    si = C.SCEN_NAMES.index(scenario)
    phi, _ = make_truth(scenario, np.random.default_rng([seed, si, real]))
    c = core.cutoff(scenario)
    truth = Truth(phi)
    ref = fitting.fit(truth.curves())
    true_azi = C.SCENARIOS[scenario]['vario']['azi']
    anis = C.SCENARIOS[scenario]['vario']['hmaj'] != C.SCENARIOS[scenario]['vario']['hmin']
    net_true = core.net_cells(phi, c)
    dv_tot, dv_net = D.drainage_value(phi), D.drainage_from_ooip(net_true)
    opt_tot = D.program_value(D.place(dv_tot, 'swap'), dv_tot)
    opt_net = D.program_value(D.place(dv_net, 'swap'), dv_net)
    field_net = net_true.sum() / 1e6
    fits, dec = [], []
    for n in C.WELL_COUNTS:
        for di, design in enumerate(C.DESIGNS):
            key = dict(scenario=scenario, realization=real, n_wells=n, design=design)
            rng = np.random.default_rng([seed, si, real, n, di, 1])
            w = W.sample(phi, design, n, rng)
            x, y, v = w['x'], w['y'], w['v']
            wt = W.decluster(x, y, v, preferential=(design == 'preferential'))
            _, var_d = W.weighted_stats(v, wt)
            ns = W.NormalScore(v, wt)
            exp = experimental(x, y, v, ns.scores)
            models = fitting.fit_all(exp)
            models['semivariogram_c'] = core.fit_constrained(exp[exp.vtype == 'semivariogram'], lag_settings(x, y)['xlag'])
            models['ml'] = core.fit_ml(x, y, ns.scores)
            models['truth_variogram'] = ref
            for meth, m in models.items():
                mm = {k: m[k] for k in KEYS}
                fits.append(dict(**key, method=meth, **mm, vario_rmse=truth.short_rmse(mm),
                                 azi_err=fitting.azimuth_error(mm, true_azi) if anis else np.nan))
                est = kriging.ordinary_kriging(x, y, v, mm, var_d)[0]
                kn, en = core.net_cells(est, c), core.etype_net(x, y, ns, mm, c)
                for rule, regret, fnet in (('map_total', _decide(C.ooip_cells(est), dv_tot, opt_tot), np.nan),
                                           ('map_net', _decide(kn, dv_net, opt_net), kn.sum() / 1e6),
                                           ('etype_net', _decide(en, dv_net, opt_net), en.sum() / 1e6)):
                    dec.append(dict(**key, method=meth, rule=rule, regret_MM=regret, field_net=fnet, true_field_net=field_net))
            idw = kriging.idw(x, y, v)
            kn = core.net_cells(idw, c)
            dec.append(dict(**key, method='idw', rule='map_total', regret_MM=_decide(C.ooip_cells(idw), dv_tot, opt_tot),
                            field_net=np.nan, true_field_net=field_net))
            dec.append(dict(**key, method='idw', rule='map_net', regret_MM=_decide(kn, dv_net, opt_net),
                            field_net=kn.sum() / 1e6, true_field_net=field_net))
            if n != 50:
                continue
            # the same decision from 50 SGS realizations (semivariogram model, 50 wells): what a simulation workflow delivers
            reals = sgs.simulate(x, y, ns, {k: models['semivariogram'][k] for k in KEYS}, N_SGS,
                                 np.random.default_rng([seed, si, real, n, di, 2]))
            net = core.net_cells(reals, c)
            tot = net.sum((1, 2)) / 1e6
            p10, p90 = np.percentile(tot, [10, 90])
            dec.append(dict(**key, method='semivariogram', rule='sgs_net', regret_MM=_decide(net.mean(0), dv_net, opt_net),
                            field_net=tot.mean(), true_field_net=field_net, net_p10=p10, net_p90=p90,
                            net_inside_p10_p90=bool(p10 <= field_net <= p90)))
    return fits, dec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    if a.pilot:
        seed, n_real, out = 555, 10, os.path.join(HERE, 'pilot_results')
    else:
        hyp = json.load(open(os.path.join(HERE, 'hypotheses.json')))
        seed, n_real, out = hyp['seed'], hyp['n_truths_per_scenario'], os.path.join(HERE, 'results')
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'
    jobs = [(s, r, seed) for r in range(n_real) for s in C.SCEN_NAMES]
    print(f'{len(jobs)} truth models, seed {seed}, {a.workers} processes', flush=True)
    t0, fits, dec = time.time(), [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (f, d) in enumerate(ex.map(run_truth, jobs), 1):
            fits += f; dec += d
            if i % max(1, len(jobs) // 10) == 0:
                print(f'  {i}/{len(jobs)} ({time.time() - t0:.0f} s)', flush=True)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(fits).to_csv(os.path.join(out, 'fits.csv'), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(out, 'decisions.csv'), index=False)
    json.dump(dict(seed=seed, n_truths_per_scenario=n_real, runtime_s=round(time.time() - t0)),
              open(os.path.join(out, 'run_info.json'), 'w'), indent=2)
    print(f'saved {out} ({time.time() - t0:.0f} s)')

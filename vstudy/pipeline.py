"""The whole workflow for one truth model, and the parallel study runner with an on-disk cache.

Every random draw comes from a generator keyed by (stage, scenario, realization[, well count,
design]), so any dataset can be rebuilt exactly in a notebook with build_dataset().
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import skew

from . import config as C
from . import crossval, decisions as D, fitting, kriging, wells as W
from .simulate import make_truth, truth_rng
from .variography import Truth, experimental

KEY = ['scenario', 'realization', 'n_wells', 'design']
MAP_METHODS = C.VTYPES + ['loocv', 'truth_variogram', 'idw']
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_truth(stage, scenario, real):
    phi, facies = make_truth(scenario, truth_rng(stage, scenario, real))
    return phi, facies


def dataset_rng(stage, scenario, real, n_wells, design):
    base = C.SEED_PILOT if stage == 'pilot' else C.SEED_MAIN
    return np.random.default_rng([base, C.SCEN_NAMES.index(scenario), real, n_wells, C.DESIGNS.index(design), 1])


def build_dataset(stage, scenario, real, n_wells, design, phi=None):
    """Wells, declustering, normal scores, experimental variograms and fitted models for one dataset."""
    phi = build_truth(stage, scenario, real)[0] if phi is None else phi
    rng = dataset_rng(stage, scenario, real, n_wells, design)
    w = W.sample(phi, design, n_wells, rng)
    w['wt'] = W.decluster(w['x'], w['y'], w['v'], preferential=(design == 'preferential'))
    w['mean_d'], w['var_d'] = W.weighted_stats(w['v'], w['wt'])
    ns = W.NormalScore(w['v'], w['wt'])
    exp = experimental(w['x'], w['y'], w['v'], ns.scores)
    models = fitting.fit_all(exp)
    return dict(wells=w, ns=ns, exp=exp, models=models, rng=rng)


def run_truth(args):
    stage, scenario, real = args
    phi, _ = build_truth(stage, scenario, real)
    truth = Truth(phi)
    ref = fitting.fit(truth.curves())
    true_ooip = float(C.ooip_total(phi))
    dv_true = D.drainage_value(phi)
    v_opt = D.program_value(D.place(dv_true, 'swap'), dv_true)
    true_azi = C.SCENARIOS[scenario]['vario']['azi']
    anis = C.SCENARIOS[scenario]['vario']['hmaj'] != C.SCENARIOS[scenario]['vario']['hmin']
    T = dict(truths=[dict(scenario=scenario, realization=real, true_ooip=true_ooip, opt_value=v_opt,
                          mean_phi=phi.mean(), std_phi=phi.std(), skew_phi=float(skew(phi.ravel())),
                          **{f'ref_{k}': v for k, v in ref.items()}, ref_vario_rmse=truth.short_rmse(ref))],
             fits=[], estimates=[], decisions=[], uncertainty=[])

    for n_wells in C.WELL_COUNTS:
        for design in C.DESIGNS:
            key = dict(scenario=scenario, realization=real, n_wells=n_wells, design=design)
            ds = build_dataset(stage, scenario, real, n_wells, design, phi)
            w, models, rng = ds['wells'], ds['models'], ds['rng']
            x, y, v = w['x'], w['y'], w['v']

            # ---- variogram models and their accuracy
            choice, loo = crossval.pick(models, x, y, v)
            for vt, m in models.items():
                T['fits'].append(dict(**key, vtype=vt, **m, vario_rmse=truth.short_rmse(m),
                                      azi_err=fitting.azimuth_error(m, true_azi) if anis else np.nan,
                                      loo_mse=loo[vt], picked=(vt == choice)))

            # ---- maps
            maps = {}
            all_models = dict(models, loocv=models[choice], truth_variogram={k: ref[k] for k in ('it', 'nug', 'azi', 'hmaj', 'hmin')})
            for meth, m in all_models.items():
                est, kvar = kriging.ordinary_kriging(x, y, v, m, w['var_d'])
                z = (est - phi) / np.sqrt(np.maximum(kvar, 1e-12))
                maps[meth] = est
                T['estimates'].append(dict(**key, method=meth, z2=float(np.mean(z ** 2)),
                                           cover90=100 * float(np.mean(np.abs(z) <= 1.645))))
            maps['idw'] = kriging.idw(x, y, v)
            T['estimates'].append(dict(**key, method='idw'))
            maps['naive'] = np.full_like(phi, w['mean_d'])
            T['estimates'].append(dict(**key, method='naive'))
            for row in T['estimates'][-len(maps):]:
                est = maps[row['method']]
                row.update(rmse_phi=float(np.sqrt(np.mean((est - phi) ** 2))),
                           ooip_err_pct=100 * (float(C.ooip_total(est)) - true_ooip) / true_ooip,
                           loocv_choice=choice if row['method'] == 'loocv' else None)

            # ---- decision: place 3 wells on each map
            for meth in MAP_METHODS:
                dv = D.drainage_value(maps[meth])
                s = D.place(dv, 'swap')
                T['decisions'].append(dict(**key, method=meth, sites=' '.join(map(str, sorted(s))),
                                           opt_value=v_opt, true_value=D.program_value(s, dv_true),
                                           est_value=D.program_value(s, dv)))

            # ---- uncertainty: conditional simulation with the normal-score model
            sims = kriging.conditional_sims(x, y, ds['ns'], models['normal_score'], C.N_SIM, rng)
            oc = C.ooip_cells(sims)
            tot = oc.sum((1, 2)) / 1e6
            p10, p50, p90 = np.percentile(tot, [10, 50, 90])
            T['uncertainty'].append(dict(**key, true_ooip=true_ooip, p10=p10, p50=p50, p90=p90,
                                         inside_p10_p90=bool(p10 <= true_ooip <= p90),
                                         krig_ooip=float(C.ooip_total(maps['normal_score']))))
            dv = D.drainage_from_ooip(kriging.expected_ooip_cells(x, y, ds['ns'], models['normal_score']))
            s = D.place(dv, 'swap')
            T['decisions'].append(dict(**key, method='simulation_ev', sites=' '.join(map(str, sorted(s))),
                                       opt_value=v_opt, true_value=D.program_value(s, dv_true),
                                       est_value=D.program_value(s, dv)))
            T['decisions'].append(dict(**key, method='random', sites='', opt_value=v_opt,
                                       true_value=D.random_programs(dv_true, C.N_RANDOM, rng), est_value=np.nan))
    return T


def _finalize(parts):
    out = {k: pd.DataFrame([r for p in parts for r in p[k]]) for k in parts[0]}
    e = out['estimates']
    e['abs_ooip_err_pct'] = e.ooip_err_pct.abs()
    ref = e[e.method == 'truth_variogram'].set_index(KEY).rmse_phi
    e['excess_rmse_pct'] = 100 * (e.rmse_phi / e.set_index(KEY).index.map(ref).values - 1)
    d = out['decisions']
    d['regret_pct'] = 100 * (d.opt_value - d.true_value) / d.opt_value
    d['regret_MM'] = C.to_money(d.opt_value - d.true_value)
    d['npv_MM'] = C.to_money(d.true_value) - C.K_WELLS * C.WELL_COST
    return out


def run_study(stage='main', workers=None, force=False, log=print):
    """Run (or load from results/<stage>/) the full study. stage = 'pilot' or 'main'."""
    folder = os.path.join(HERE, 'results', stage)
    names = ['truths', 'fits', 'estimates', 'decisions', 'uncertainty']
    if not force and all(os.path.exists(os.path.join(folder, f'{n}.csv')) for n in names):
        return {n: pd.read_csv(os.path.join(folder, f'{n}.csv')) for n in names}
    n_real = C.N_REAL_PILOT if stage == 'pilot' else C.N_REAL
    jobs = [(stage, s, r) for s in C.SCEN_NAMES for r in range(n_real)]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    t0, parts = time.time(), []
    log(f'{stage}: {len(jobs)} truth models x {len(C.WELL_COUNTS) * len(C.DESIGNS)} well sets on {workers} processes')
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'                                              # one BLAS thread per worker
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, p in enumerate(ex.map(run_truth, jobs), 1):
            parts.append(p)
            if i % max(1, len(jobs) // 8) == 0 or i == len(jobs):
                log(f'  {i}/{len(jobs)} done ({time.time() - t0:.0f} s)')
    out = _finalize(parts)
    os.makedirs(folder, exist_ok=True)
    for n, df in out.items():
        df.to_csv(os.path.join(folder, f'{n}.csv'), index=False)
    return out

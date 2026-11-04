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
from . import crossval, decisions as D, fitting, kriging, sgs, wells as W
from .areas import STUDY_AOIS, LOCAL
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
             fits=[], estimates=[], decisions=[], uncertainty=[], simulation=[], aoi=[], aoi_decisions=[])
    true_aoi = {a.name: (float(a.mean(phi)), float(a.ooip(phi))) for a in STUDY_AOIS}

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

            # ---- volumetrics in the fixed areas of interest, and which local AOI to develop
            for meth in MAP_METHODS + ['naive']:
                est = {a.name: float(a.ooip(maps[meth])) for a in STUDY_AOIS}
                for a in STUDY_AOIS:
                    T['aoi'].append(dict(**key, method=meth, source='map', aoi=a.name, true_phi=true_aoi[a.name][0],
                                         true_ooip=true_aoi[a.name][1], est_phi=float(a.mean(maps[meth])), est_ooip=est[a.name]))
                if meth != 'naive':
                    T['aoi_decisions'].append(aoi_choice(key, meth, 'kriged map', {n: est[n] for n in C.AOI_CHOICE}, true_aoi))

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

            # ---- estimation by sequential Gaussian simulation, one set of realizations per variogram model
            sim_rows, sim_dec, aoi_rows, aoi_dec = simulate_and_decide(stage, key, x, y, ds['ns'], all_models, choice, phi,
                                                                       dv_true, v_opt, true_ooip, true_aoi)
            T['simulation'] += sim_rows
            T['decisions'] += sim_dec
            T['aoi'] += aoi_rows
            T['aoi_decisions'] += aoi_dec
    return T


SIM_TAGS = {'sgs_ev': None, 'sgs_p10': C.RISK_Q}      # decision rules on the realizations


def aoi_choice(key, method, rule, score, true_aoi):
    """Develop the local AOI with the highest score (estimated OOIP, its mean or its P10).
    Regret = true OOIP of the best local AOI minus that of the chosen one."""
    chosen = max(score, key=score.get)
    best = max(C.AOI_CHOICE, key=lambda n: true_aoi[n][1])
    return dict(**key, method=method, rule=rule, chosen=chosen, best=best, true_chosen=true_aoi[chosen][1],
                true_best=true_aoi[best][1], regret_mmstb=true_aoi[best][1] - true_aoi[chosen][1])


def simulate_and_decide(stage, key, x, y, ns, all_models, choice, phi, dv_true, v_opt, true_ooip, true_aoi):
    """N_SGS realizations per variogram model (the 8 types and the true variogram; the cross-validation
    pick reuses its type's). Decisions (PGE 338 Lecture 16c): sgs_ev places the wells that maximize the
    expected drained OOIP over the realizations; sgs_p10 the wells that maximize its P10 (risk-averse).
    Each model's fitted standardized variogram is used for the declustered normal scores."""
    base = C.SEED_PILOT if stage == 'pilot' else C.SEED_MAIN
    ids = [C.SCEN_NAMES.index(key['scenario']), key['realization'], key['n_wells'], C.DESIGNS.index(key['design'])]
    rows, dec, aoi_rows, aoi_dec = [], [], [], []
    for mi, meth in enumerate(C.VTYPES + ['truth_variogram']):
        reals = sgs.simulate(x, y, ns, all_models[meth], C.N_SGS, np.random.default_rng([base, *ids, 2, mi]))
        vals = np.array([D.drainage_from_ooip(C.ooip_cells(r)).ravel()[D.CAND] for r in reals])   # (L, n_cand)
        tot = C.ooip_total(reals)
        p10, p90 = np.percentile(tot, [10, 90])
        rows.append(dict(**key, method=meth, ooip_p10=p10, ooip_p50=np.percentile(tot, 50), ooip_p90=p90,
                         true_ooip=true_ooip, ooip_inside_p10_p90=bool(p10 <= true_ooip <= p90),
                         sd_field_mean_phi=float(reals.mean((1, 2)).std(ddof=1)),
                         rmse_phi_realization=float(np.sqrt(np.mean((reals - phi) ** 2)))))
        for tag, q in SIM_TAGS.items():
            s = D.greedy(vals.mean(0)) if q is None else None
            s = D.improve(vals.mean(0), s) if q is None else D.place_quantile(vals, q)
            prog = vals[:, s].sum(1)
            tv = D.program_value(s, dv_true)
            lo, hi = np.percentile(prog, [10, 90])
            for name in [meth] + (['loocv'] if meth == choice else []):
                dec.append(dict(**key, method=f'{name}@{tag}', sites=' '.join(map(str, sorted(s))), opt_value=v_opt,
                                true_value=tv, est_value=float(prog.mean()), est_p10=lo, est_p90=hi,
                                inside_p10_p90=bool(lo <= tv <= hi)))
        # AOI volumetrics from the realizations: distribution of OOIP in each fixed AOI
        a_oo = {a.name: a.ooip(reals) for a in STUDY_AOIS}                # name -> (L,) MMSTB
        for a in STUDY_AOIS:
            o = a_oo[a.name]
            lo, hi = np.percentile(o, [10, 90])
            aoi_rows.append(dict(**key, method=meth, source='SGS', aoi=a.name, true_phi=true_aoi[a.name][0],
                                 true_ooip=true_aoi[a.name][1], est_phi=float(a.mean(reals).mean()), est_ooip=float(o.mean()),
                                 p10_ooip=lo, p90_ooip=hi, inside_p10_p90=bool(lo <= true_aoi[a.name][1] <= hi)))
        for tag, q in SIM_TAGS.items():
            score = {n: float(a_oo[n].mean() if q is None else np.quantile(a_oo[n], q)) for n in C.AOI_CHOICE}
            for name in [meth] + (['loocv'] if meth == choice else []):
                aoi_dec.append(aoi_choice(key, name, tag, score, true_aoi))
    return rows, dec, aoi_rows, aoi_dec


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
    if 'aoi' in out and len(out['aoi']):
        a = out['aoi']
        a['err_pct'] = 100 * (a.est_ooip - a.true_ooip) / a.true_ooip
        a['abs_err_pct'] = a.err_pct.abs()
        out['aoi_decisions']['regret_MM'] = C.to_money(out['aoi_decisions'].regret_mmstb)
    return out


def run_study(stage='main', workers=None, force=False, log=print):
    """Run (or load from results/<stage>/) the full study. stage = 'pilot' or 'main'."""
    folder = os.path.join(HERE, 'results', stage)
    names = ['truths', 'fits', 'estimates', 'decisions', 'uncertainty']
    if not force and all(os.path.exists(os.path.join(folder, f'{n}.csv')) for n in names):
        extra = [n for n in ['simulation', 'aoi', 'aoi_decisions'] if os.path.exists(os.path.join(folder, f'{n}.csv'))]
        return {n: pd.read_csv(os.path.join(folder, f'{n}.csv')) for n in names + extra}
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

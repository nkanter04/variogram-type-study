"""Area-of-interest (AOI) volumetrics and variogram sensitivity with sequential Gaussian simulation.

A volumetric decision is made on the average porosity / OOIP inside an AOI. Kriging gives the best
single estimate of that average but compresses variance; SGS realizations add the spatial
(ergodic) fluctuation, so the AOI average varies from realization to realization. How much it
varies, and where it centres, depends on the variogram: over the whole field the variogram effect
averages out, over a small AOI between a high and a low well the range and azimuth decide which
well dominates.

Demonstration case: anisotropic scenario (true azimuth 045, ranges 600 / 150 m), truth 0,
25 random wells. AOIs are axis-aligned rectangles, i.e. NumPy slices of the grid.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import config as C, kriging, sgs
from .areas import AOI
from .pipeline import build_truth, build_dataset, HERE

CASE = dict(stage='main', scenario='anisotropic', real=0, n_wells=25, design='random')
N_SIM = 100                      # realizations per variogram model (distribution of AOI means)
AOI_SIZE = 20                    # local AOI side in cells (200 m x 200 m, about one well spacing)
RANGES = [100.0, 200.0, 400.0, 800.0]   # isotropic ranges for the range-sensitivity sweep
FOLDER = os.path.join(HERE, 'results', 'aoi')


GLOBAL = AOI('global (whole field)', slice(0, C.NY), slice(0, C.NX))


# ------------------------------------------------------------------ case and models
def case():
    phi = build_truth(CASE['stage'], CASE['scenario'], CASE['real'])[0]
    ds = build_dataset(CASE['stage'], CASE['scenario'], CASE['real'], CASE['n_wells'], CASE['design'], phi)
    return phi, ds


def models(ds):
    """Variogram models compared (all sill 1, used for the normal scores)."""
    true = dict(C.SCENARIOS[CASE['scenario']]['vario'])
    key = lambda m: {k: m[k] for k in ('it', 'nug', 'azi', 'hmaj', 'hmin')}
    out = {'correct azimuth (045)': true,
           'rotated 90 deg (135)': dict(true, azi=(true['azi'] + 90.0) % 180.0),
           'fitted from wells (normal score)': key(ds['models']['normal_score'])}
    for vt in C.VTYPES:
        out[f'type: {vt}'] = key(ds['models'][vt])
    for a in RANGES:
        out[f'isotropic range {a:.0f} m'] = dict(it=1, nug=true['nug'], azi=0.0, hmaj=a, hmin=a)
    return out


DEMO = ['correct azimuth (045)', 'rotated 90 deg (135)', 'fitted from wells (normal score)']


# ------------------------------------------------------------------ AOI selection
def _window_sums(a, s):
    c = np.pad(np.cumsum(np.cumsum(a, 0), 1), ((1, 0), (1, 0)))
    return c[s:, s:] - c[:-s, s:] - c[s:, :-s] + c[:-s, :-s]


def pick_aois(ds, mods, size=AOI_SIZE):
    """Global AOI plus two local AOIs chosen from the kriged maps of the correct and rotated models:
    'variogram-sensitive' = the window where the two kriged averages differ most;
    'azimuth-insensitive' = the window containing a well where they differ least (not overlapping).
    The local AOIs are picked to illustrate the effect, which a real study would not do: there the
    AOI is fixed by decision makers before modelling."""
    w = ds['wells']
    ka = kriging.ordinary_kriging(w['x'], w['y'], w['v'], mods[DEMO[0]], w['var_d'])[0]
    kb = kriging.ordinary_kriging(w['x'], w['y'], w['v'], mods[DEMO[1]], w['var_d'])[0]
    diff = np.abs(_window_sums(ka, size) - _window_sums(kb, size)) / size ** 2
    r, c = np.unravel_index(np.argmax(diff), diff.shape)
    sens = AOI('local, variogram-sensitive', slice(r, r + size), slice(c, c + size))
    occ = np.zeros((C.NY, C.NX)); occ[w['row'], w['col']] = 1
    has_well = _window_sums(occ, size) > 0
    rr, cc = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
    apart = (np.abs(rr - r) >= size) | (np.abs(cc - c) >= size)
    d2 = np.where(has_well & apart, diff, np.inf)
    r2, c2 = np.unravel_index(np.argmin(d2), d2.shape)
    ctrl = AOI('local, azimuth-insensitive', slice(r2, r2 + size), slice(c2, c2 + size))
    return [GLOBAL, sens, ctrl]


# ------------------------------------------------------------------ simulation runner
def _job(args):
    name, m, n_sim, seed, aois, keep = args
    phi, ds = case()
    w = ds['wells']
    reals = sgs.simulate(w['x'], w['y'], ds['ns'], m, n_sim, np.random.default_rng(seed))
    krig = kriging.ordinary_kriging(w['x'], w['y'], w['v'], m, w['var_d'])[0]
    rows = []
    for a in aois:
        pm, om = a.mean(reals), a.ooip(reals)
        rows.append(dict(model=name, aoi=a.name, truth_phi=a.mean(phi), truth_ooip=a.ooip(phi),
                         krig_phi=a.mean(krig), krig_ooip=a.ooip(krig),
                         sgs_mean_phi=pm.mean(), sgs_sd_phi=pm.std(ddof=1),
                         p10_phi=np.percentile(pm, 10), p50_phi=np.percentile(pm, 50), p90_phi=np.percentile(pm, 90),
                         sgs_mean_ooip=om.mean(), p10_ooip=np.percentile(om, 10), p90_ooip=np.percentile(om, 90),
                         truth_in_p10_p90=bool(np.percentile(pm, 10) <= a.mean(phi) <= np.percentile(pm, 90))))
    if name.startswith('isotropic'):                                      # same range, no wells: pure ergodic fluctuation
        u = ds['ns'].back(sgs.simulate_scores(np.array([]), np.array([]), np.array([]), m, n_sim, np.random.default_rng(seed + 500)))
        rows[0]['uncond_sd_phi'] = float(GLOBAL.mean(u).std(ddof=1))
    means = np.array([[a.mean(r) for a in aois] for r in reals])          # (n_sim, n_aoi)
    return name, rows, means, (reals[:3].astype(np.float32) if keep else None), krig.astype(np.float32)


def run(force=False, workers=None, log=print):
    """Run (or load from results/aoi/) the AOI study. Returns dict(aois, summary, means, reals, krig, phi, ds, models)."""
    phi, ds = case()
    mods = models(ds)
    files = [os.path.join(FOLDER, f) for f in ('aois.json', 'summary.csv', 'sims.npz')]
    if force or not all(os.path.exists(f) for f in files):
        aois = pick_aois(ds, mods)
        jobs = [(k, m, N_SIM, 1000 + i, aois, k in DEMO) for i, (k, m) in enumerate(mods.items())]
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[var] = '1'
        workers = workers or max(1, (os.cpu_count() or 2) - 2)
        log(f'SGS: {len(jobs)} variogram models x {N_SIM} realizations on {workers} processes')
        rows, store = [], {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for name, r, means, reals, krig in ex.map(_job, jobs):
                rows += r
                store[f'means|{name}'] = means
                store[f'krig|{name}'] = krig
                if reals is not None:
                    store[f'reals|{name}'] = reals
        os.makedirs(FOLDER, exist_ok=True)
        json.dump([a.to_dict() for a in aois], open(files[0], 'w'), indent=2)
        pd.DataFrame(rows).to_csv(files[1], index=False)
        np.savez_compressed(files[2], **store)
    aois = [AOI.from_dict(d) for d in json.load(open(files[0]))]
    z = np.load(files[2])
    get = lambda p: {k.split('|', 1)[1]: z[k] for k in z.files if k.startswith(p + '|')}
    return dict(aois=aois, summary=pd.read_csv(files[1]), means=get('means'), reals=get('reals'), krig=get('krig'),
                phi=phi, ds=ds, models=mods)


# ------------------------------------------------------------------ figures
def show(ax, field, aoi=None, wells=None, vmin=0.08, vmax=0.26, title=''):
    """Field semi-transparent outside the AOI, opaque inside, AOI outlined."""
    ext = [C.XMIN, C.XMAX, C.YMIN, C.YMAX]
    local = aoi is not None and aoi.name != GLOBAL.name
    ax.imshow(field, extent=ext, cmap='inferno', vmin=vmin, vmax=vmax, alpha=0.35 if local else 1.0, interpolation='none')
    if local:
        inside = np.full(field.shape, np.nan); inside[aoi.rows, aoi.cols] = field[aoi.rows, aoi.cols]
        im = ax.imshow(inside, extent=ext, cmap='inferno', vmin=vmin, vmax=vmax, interpolation='none')
        x0, x1, y0, y1 = aoi.extent()
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='white', lw=1.6))
    if wells is not None:
        ax.scatter(wells['x'], wells['y'], c=wells['v'], cmap='inferno', vmin=vmin, vmax=vmax, s=30,
                   edgecolors='white', linewidths=0.9, zorder=3)
    ax.set_xticks([0, 500, 1000]); ax.set_yticks([0, 500, 1000])
    ax.set_title(title, fontsize=9)


def realization_grid(R, aoi):
    """3 x 3: rows = demo variogram models, columns = realizations 1-3; AOI average in each title."""
    fig, axes = plt.subplots(3, 3, figsize=(11, 11.4))
    w = R['ds']['wells']
    for i, name in enumerate(DEMO):
        for j in range(3):
            f = R['reals'][name][j]
            show(axes[i, j], f, aoi, w, title=f'{name}\nrealization {j + 1}: AOI mean phi = {aoi.mean(f):.4f}')
    fig.suptitle(f"{aoi.name} AOI; true AOI mean phi = {aoi.mean(R['phi']):.4f}", y=0.995, fontsize=11)
    fig.tight_layout()
    return fig


def histograms(R, names=DEMO, colors=('#1f6fc4', '#eb6834', '#1baf7a')):
    """Distribution of AOI mean porosity over the realizations, per AOI; truth (black) and kriging (dashed)."""
    fig, axes = plt.subplots(1, len(R['aois']), figsize=(5.1 * len(R['aois']), 3.5))
    for k, (ax, a) in enumerate(zip(axes, R['aois'])):
        allv = np.concatenate([R['means'][n][:, k] for n in names])
        bins = np.linspace(allv.min(), allv.max(), 26)
        for n, col in zip(names, colors):
            ax.hist(R['means'][n][:, k], bins=bins, color=col, alpha=0.45, label=n)
            ax.axvline(a.mean(R['krig'][n]), color=col, ls='--', lw=1.5)
        ax.axvline(a.mean(R['phi']), color='black', lw=2, label='truth')
        ax.set_title(a.name, fontsize=10); ax.set_xlabel('AOI mean porosity')
    axes[0].set_ylabel('realizations'); axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    return fig

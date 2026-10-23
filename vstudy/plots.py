"""Figures with a fixed colour scheme: major = blue, minor = orange, truth = dark grey,
sequential blue heatmaps. One function per figure type used in the notebooks."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config as C
from .models import gamma_along

C_MAJ, C_MIN, C_TRUTH, C_MUTED = '#2a78d6', '#eb6834', '#3a3a38', '#9a9993'
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
PHI_CMAP = plt.cm.inferno
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': False, 'font.size': 9.5})
EXTENT = [C.XMIN, C.XMAX, C.YMIN, C.YMAX]


def lab(k):
    return C.LABEL.get(k, k)


def maps(panels, wells=None, ncols=3, vmin=None, vmax=None, cmap=PHI_CMAP, cbar='Porosity (fraction)', size=4.4):
    """panels: list of (title, 2-D array)."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows * 0.92), squeeze=False)
    vmin = np.percentile(panels[0][1], 1) if vmin is None else vmin
    vmax = np.percentile(panels[0][1], 99) if vmax is None else vmax
    for ax, (title, a) in zip(axes.ravel(), panels):
        im = ax.imshow(a, extent=EXTENT, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        if wells is not None:
            ax.scatter(wells['x'], wells['y'], s=12, facecolors='none', edgecolors='white', linewidths=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 500, 1000]); ax.set_yticks([0, 500, 1000])
    for ax in axes.ravel()[len(panels):]:
        ax.axis('off')
    fig.colorbar(im, ax=axes, shrink=0.75, label=cbar)
    return fig


def variogram_panels(exp, models, truth_curves, types=C.VTYPES, ncols=4):
    """Experimental major/minor curves (dots sized by pairs), fitted model (line) and truth (dashed)."""
    nrows = int(np.ceil(len(types) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.4 * nrows), sharex=True, sharey=True, squeeze=False)
    hh = np.linspace(1, C.MAX_LAG, 200)
    for ax, vt in zip(axes.ravel(), types):
        m = models[vt]
        for name, azi, col, mk in (('major', m['azi'], C_MAJ, 'o'), ('minor', (m['azi'] + 90) % 180, C_MIN, '^')):
            e = exp[(exp.vtype == vt) & (exp.azi == azi) & (exp.npairs >= C.MIN_PAIRS)]
            t = truth_curves[truth_curves.azi == azi]
            ax.plot(t.h, t.std_value, '--', color=col, lw=1, alpha=0.75)
            if vt == 'cumulative':
                dh = np.diff(np.r_[0.0, e.h.values])
                ax.plot(e.h, e.std_value, mk, color=col, ms=5, mec='white')
                ax.plot(e.h, np.cumsum(gamma_along(e.h.values, azi, m) * dh) / e.h.values, '-', color=col, lw=2,
                        label=f'{name} {azi:.0f}°')
            else:
                ax.scatter(e.h, e.std_value, s=6 + 1.5 * e.npairs, marker=mk, color=col, edgecolors='white', linewidths=0.6, zorder=3)
                ax.plot(hh, gamma_along(hh, azi, m), '-', color=col, lw=2, label=f'{name} {azi:.0f}°')
        ax.axhline(1, color=C_MUTED, lw=0.8)
        ax.set_title(f"{lab(vt)}\nnug {m['nug']:.2f}, {m['hmaj']:.0f} / {m['hmin']:.0f} m", fontsize=9.5)
        ax.set_ylim(0, 1.8); ax.set_xlim(0, C.MAX_LAG)
        ax.legend(fontsize=7.5, loc='lower right', frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel('Lag (m)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Standardized value')
    fig.tight_layout()
    return fig


def heat(tab, title, fmt='{:.2f}', cbar=''):
    """Sequential heatmap with values printed; the lowest value in each row is boxed."""
    v = tab.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.05 * v.shape[1] + 2.6, 0.45 * v.shape[0] + 1.6))
    lo, hi = np.nanmin(v), np.nanmax(v)
    im = ax.imshow(v, cmap=SEQ, aspect='auto', vmin=lo, vmax=hi)
    for i in range(v.shape[0]):
        jb = int(np.nanargmin(v[i]))
        for j in range(v.shape[1]):
            ax.text(j, i, fmt.format(v[i, j]), ha='center', va='center', fontsize=8.5,
                    color='white' if (v[i, j] - lo) > 0.55 * (hi - lo) else '#0b0b0b',
                    fontweight='bold' if j == jb else 'normal')
        ax.add_patch(plt.Rectangle((jb - 0.5, i - 0.5), 1, 1, fill=False, ec='#0b0b0b', lw=1.6))
    ax.set_xticks(range(v.shape[1])); ax.set_xticklabels([lab(c) for c in tab.columns], rotation=30, ha='right')
    ax.set_yticks(range(v.shape[0])); ax.set_yticklabels([str(i) for i in tab.index])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, label=cbar, shrink=0.85)
    ax.set_title(title, loc='left')
    fig.tight_layout()
    return fig


def forest(tables, xlabel, title):
    """tables: {panel title: paired_table}. Mean difference vs semivariogram with 95% CI."""
    fig, axes = plt.subplots(1, len(tables), figsize=(4.6 * len(tables), 0.36 * len(next(iter(tables.values()))) + 1.6),
                             sharey=True, squeeze=False)
    for ax, (name, t) in zip(axes[0], tables.items()):
        y = np.arange(len(t))[::-1]
        ax.axvline(0, color=C_MUTED, lw=1)
        sig = (t.p_holm < 0.05).values
        ax.hlines(y, t.ci_lo, t.ci_hi, color=C_MAJ, lw=2)
        ax.scatter(t.mean_diff, y, s=40, color=np.where(sig, C_MIN, C_MAJ), edgecolors='white', zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([lab(m) for m in t.index])
        ax.set_title(name, fontsize=10); ax.set_xlabel(xlabel)
    fig.suptitle(title + '   (orange = significant after Holm correction)', x=0.01, ha='left', fontsize=10.5)
    fig.tight_layout()
    return fig


def value_bars(voi, title):
    """voi: analysis.value_of_information table. Bars = mean $MM, whiskers = 95% CI; negatives orange."""
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(voi) + 1.3))
    y = np.arange(len(voi))[::-1]
    v = voi.value_MM.values
    ax.barh(y, v, color=[C_MAJ if x >= 0 else C_MIN for x in v], height=0.55)
    ax.errorbar(v, y, xerr=[v - voi.ci_lo.values, voi.ci_hi.values - v], fmt='none', ecolor=C_TRUTH, lw=1, capsize=3)
    for yi, x, hi in zip(y, v, voi.ci_hi.values):
        ax.text(max(hi, 0) + 0.1, yi, f"{'+' if x >= 0 else '-'}${abs(x):.2f}MM", va='center', fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(voi.index)
    ax.axvline(0, color=C_MUTED, lw=1)
    ax.set_xlabel('Value per 3-well program ($MM), 95% CI clustered by truth model')
    ax.set_title(title, loc='left')
    ax.set_xlim(min(0, voi.ci_lo.min()) - 0.3, voi.ci_hi.max() * 1.18 + 0.3)
    fig.tight_layout()
    return fig

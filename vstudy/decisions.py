"""Development decisions (PGE 338 Lecture 16c framing: choose the scenario that maximizes value).

Decision 1: place K_WELLS wells, each draining the OOIP within R_DRAIN, at least 2*R_DRAIN apart.
Decision 2: go/no-go on total OOIP against a threshold.
"""
import numpy as np
from scipy.signal import fftconvolve

from . import config as C

_rc = int(np.ceil(C.R_DRAIN / C.XSIZ))
_dy, _dx = np.mgrid[-_rc:_rc + 1, -_rc:_rc + 1]
DISK = ((_dx * C.XSIZ) ** 2 + (_dy * C.YSIZ) ** 2 <= C.R_DRAIN ** 2).astype(float)
_cand = np.zeros((C.NY, C.NX), bool)
_cand[C.CAND_STEP // 2::C.CAND_STEP, C.CAND_STEP // 2::C.CAND_STEP] = True
CAND = np.flatnonzero(_cand.ravel())
CX, CY = C.XX.ravel()[CAND], C.YY.ravel()[CAND]


def drainage_from_ooip(ooip_cells):
    """MMSTB drained by a well at each cell (cells outside the field contribute nothing)."""
    return np.maximum(fftconvolve(ooip_cells, DISK, mode='same'), 0.0) / 1e6


def drainage_value(phi):
    return drainage_from_ooip(C.ooip_cells(phi))


def _ok_with(i, chosen):
    return all(np.hypot(CX[i] - CX[j], CY[i] - CY[j]) >= 2 * C.R_DRAIN for j in chosen)


def greedy(values):
    """values: drainage value at each candidate (len(CAND)). Returns candidate indices."""
    chosen = []
    for i in np.argsort(values)[::-1]:
        if _ok_with(i, chosen):
            chosen.append(int(i))
            if len(chosen) == C.K_WELLS:
                break
    return chosen


def improve(values, chosen):
    """1-swap local search: replace one well at a time by the best feasible candidate until no gain.
    Makes the 'truth-optimal' benchmark closer to the real optimum than greedy alone."""
    chosen = list(chosen)
    while True:
        best_gain, best = 1e-12, None
        for k in range(len(chosen)):
            others = chosen[:k] + chosen[k + 1:]
            feas = np.ones(len(CAND), bool)
            for j in others:
                feas &= np.hypot(CX - CX[j], CY - CY[j]) >= 2 * C.R_DRAIN
            feas[chosen] = False
            if not feas.any():
                continue
            i = int(np.argmax(np.where(feas, values, -np.inf)))
            gain = values[i] - values[chosen[k]]
            if gain > best_gain:
                best_gain, best = gain, (k, i)
        if best is None:
            return chosen
        chosen[best[0]] = best[1]


def place(dv_map, optimizer='greedy'):
    vals = dv_map.ravel()[CAND]
    s = greedy(vals)
    return improve(vals, s) if optimizer == 'swap' else s


def program_value(sites, dv_map):
    return float(dv_map.ravel()[CAND][sites].sum())


def random_programs(dv_true, n, rng):
    vals = dv_true.ravel()[CAND]
    out = []
    for _ in range(n):
        chosen = []
        for i in rng.permutation(len(CAND)):
            if _ok_with(i, chosen):
                chosen.append(int(i))
                if len(chosen) == C.K_WELLS:
                    break
        out.append(vals[chosen].sum())
    return float(np.mean(out))

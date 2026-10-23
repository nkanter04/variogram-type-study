"""Leave-one-out cross-validation of ordinary kriging in closed form (Dubrule 1983).

With A the OK matrix [[K, 1], [1', 0]] and b = [z, 0], the error of re-estimating datum i from
the other n-1 data is (A^-1 b)_i / (A^-1)_ii: one matrix inverse instead of n kriging systems.
"""
import numpy as np

from .models import covariance


def loo_errors(x, y, v, m):
    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    n = len(v)
    A = np.ones((n + 1, n + 1)); A[n, n] = 0.0
    A[:n, :n] = covariance(x[:, None] - x[None, :], y[:, None] - y[None, :], m)
    Ai = np.linalg.inv(A)
    return (Ai @ np.r_[v, 0.0])[:n] / np.diag(Ai)[:n]


def loo_mse(x, y, v, m):
    return float(np.mean(loo_errors(x, y, v, m) ** 2))


def pick(models, x, y, v):
    """Type whose model re-predicts the wells best; ties go to the earlier type in the list."""
    scores = {vt: loo_mse(x, y, v, m) for vt, m in models.items()}
    return min(scores, key=scores.get), scores

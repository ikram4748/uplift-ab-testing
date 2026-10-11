"""Planning an experiment, and why you must not stop it early.

    n_per_arm_proportion() / n_per_arm_mean()   sample size for a two-sided test (normal approximation)
    mde_proportion()                            the smallest effect a given sample size can detect
    peeking_simulation()                        A/A tests checked repeatedly: how often "significant" by chance
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _z(alpha: float, power: float) -> float:
    return stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)


def n_per_arm_proportion(p0: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Customers per arm to detect an absolute lift `mde_abs` on a baseline rate `p0`."""
    p1 = p0 + mde_abs
    var = p0 * (1 - p0) + p1 * (1 - p1)
    return int(np.ceil(_z(alpha, power) ** 2 * var / mde_abs ** 2))


def n_per_arm_mean(sd: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    return int(np.ceil(2 * _z(alpha, power) ** 2 * sd ** 2 / mde ** 2))


def mde_proportion(p0: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute lift detectable with `n_per_arm` customers per arm (solved numerically)."""
    lo, hi = 1e-6, 1 - p0 - 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if n_per_arm_proportion(p0, mid, alpha, power) > n_per_arm else (lo, mid)
    return hi


def peeking_simulation(p: float = 0.1, n_per_arm: int = 20_000, looks: int = 10, sims: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> dict:
    """A/A tests (no real effect). Each is checked `looks` times at equal intervals with an ordinary z-test.
    Returns the false-positive rate if you only look at the end, and if you stop at the first significant look."""
    rng = np.random.default_rng(seed)
    step = n_per_arm // looks
    a = rng.binomial(1, p, size=(sims, looks * step)).reshape(sims, looks, step).sum(axis=2).cumsum(axis=1)
    b = rng.binomial(1, p, size=(sims, looks * step)).reshape(sims, looks, step).sum(axis=2).cumsum(axis=1)
    n = step * np.arange(1, looks + 1)
    pa, pb = a / n, b / n
    pooled = (a + b) / (2 * n)
    z = (pa - pb) / np.sqrt(pooled * (1 - pooled) * 2 / n)
    sig = np.abs(z) > stats.norm.ppf(1 - alpha / 2)
    first = np.where(sig.any(axis=1), sig.argmax(axis=1), -1)
    return {
        "fixed_horizon_fpr": float(sig[:, -1].mean()),
        "peeking_fpr": float(sig.any(axis=1).mean()),
        "fpr_by_looks": [float(sig[:, : k + 1].any(axis=1).mean()) for k in range(looks)],
        "stopped_at_look": np.bincount(first[first >= 0], minlength=looks).tolist(),
        "looks": looks, "sims": sims, "alpha": alpha,
    }

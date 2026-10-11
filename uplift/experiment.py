"""Reading out a randomized experiment: is it valid, and how big is the effect?

    srm()            sample-ratio mismatch: did randomization produce the planned group sizes?
    balance()        standardized mean differences of pre-treatment covariates between arms
    diff_in_means()  the plain estimate, Welch standard error
    cuped()          CUPED: subtract the part of the outcome a pre-period covariate predicts (less variance,
                     same expected effect, so narrower confidence intervals for free)
    regression_adjusted()  the multi-covariate generalisation of CUPED (OLS with robust standard errors)
    bootstrap_ci()   percentile bootstrap, for heavy-tailed outcomes like spend
    holm()           Holm-Bonferroni correction across the arm x metric comparisons
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

Z95 = stats.norm.ppf(0.975)


@dataclass
class Estimate:
    effect: float
    se: float
    lo: float
    hi: float
    p: float
    control_mean: float

    @property
    def relative(self) -> float:
        return self.effect / self.control_mean if self.control_mean else float("nan")


def _estimate(effect: float, se: float, control_mean: float) -> Estimate:
    p = 2 * stats.norm.sf(abs(effect / se)) if se > 0 else float("nan")
    return Estimate(effect, se, effect - Z95 * se, effect + Z95 * se, p, control_mean)


def srm(counts: dict[str, int], expected: dict[str, float] | None = None) -> tuple[float, float]:
    """Chi-square goodness of fit of group sizes against the planned split. Returns (chi2, p).
    A tiny p (< 0.001 is the usual alarm) means the assignment or the logging is broken: stop and debug."""
    names = list(counts)
    obs = np.array([counts[n] for n in names], dtype=float)
    shares = np.array([(expected or {}).get(n, 1 / len(names)) for n in names])
    chi2, p = stats.chisquare(obs, obs.sum() * shares / shares.sum())
    return float(chi2), float(p)


def balance(df: pd.DataFrame, covariates: list[str], arm_col: str = "arm", control: str = "control") -> pd.DataFrame:
    """Standardized mean difference (treated - control) / pooled sd per covariate and arm. |SMD| < 0.1 is the
    usual threshold for "balanced". Categorical columns are one-hot encoded first."""
    x = pd.get_dummies(df[covariates], dtype=float)
    c = x[df[arm_col] == control]
    rows = {}
    for arm in sorted(set(df[arm_col]) - {control}):
        t = x[df[arm_col] == arm]
        pooled = np.sqrt((t.var() + c.var()) / 2).replace(0, np.nan)
        rows[arm] = (t.mean() - c.mean()) / pooled
    return pd.DataFrame(rows)


def diff_in_means(y_t: np.ndarray, y_c: np.ndarray) -> Estimate:
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    return _estimate(y_t.mean() - y_c.mean(), se, y_c.mean())


def cuped(y: np.ndarray, x: np.ndarray, treated: np.ndarray) -> tuple[Estimate, float, float]:
    """CUPED (Deng et al., 2013). theta = cov(y, x) / var(x), estimated on both arms pooled; the adjusted outcome
    y - theta * (x - mean(x)) has the same expected difference between arms but less variance, because x was
    measured before treatment and so is independent of it. Returns (estimate, theta, variance reduction)."""
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    y_adj = y - theta * (x - x.mean())
    reduction = 1 - y_adj.var(ddof=1) / y.var(ddof=1)
    est = diff_in_means(y_adj[treated], y_adj[~treated])
    est.control_mean = y[~treated].mean()
    return est, float(theta), float(reduction)


def regression_adjusted(y: np.ndarray, treated: np.ndarray, covariates: pd.DataFrame) -> Estimate:
    """OLS of y on the treatment indicator plus centred covariates, HC1 (heteroskedasticity-robust) standard errors.
    With randomization this is unbiased for the average effect, and usually tighter than CUPED with one covariate."""
    xc = covariates - covariates.mean()
    design = sm.add_constant(pd.concat([pd.Series(treated.astype(float), index=xc.index, name="t"), xc], axis=1))
    fit = sm.OLS(y, design).fit(cov_type="HC1")
    return _estimate(float(fit.params["t"]), float(fit.bse["t"]), float(y[~treated].mean()))


def bootstrap_ci(y_t: np.ndarray, y_c: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.array([rng.choice(y_t, len(y_t)).mean() - rng.choice(y_c, len(y_c)).mean() for _ in range(reps)])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate, uniformly more powerful than
    plain Bonferroni)."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()

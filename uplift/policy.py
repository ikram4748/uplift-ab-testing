"""From effects to a decision: which e-mail, if any, should each customer get?

A *policy* maps each customer to an arm (control = no e-mail, mens, womens). Its value is estimated on held-out
customers without running a new experiment, by inverse-propensity weighting (IPW): customers whose random
assignment happens to match what the policy would have chosen stand in for everyone the policy would treat
that way. This is unbiased here because the propensities are known (each arm had a 1/3 chance).

Profit per customer = margin x spend - cost of the e-mail (if one is sent). The margin and the cost are business
assumptions, so every result is reported for several cost levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .models import PARAMS

ARMS = ["control", "mens", "womens"]


def profit(spend: np.ndarray, emailed: np.ndarray, margin: float, cost: float) -> np.ndarray:
    return margin * spend - cost * emailed


def value(assign: np.ndarray, arm: np.ndarray, spend: np.ndarray, margin: float, cost: float,
          shares: dict[str, float]) -> float:
    """Self-normalised IPW estimate of average profit per customer if `assign` were used (Hajek estimator:
    lower variance than plain IPW, and it cannot go outside the range of observed profits)."""
    match = assign == arm
    w = match / np.array([shares[a] for a in arm])
    prof = profit(spend, assign != "control", margin, cost)
    return float((w * prof).sum() / w.sum())


class ArmModels:
    """One outcome model per arm (a 3-arm T-learner). Predicts expected outcome under each arm."""

    def fit(self, x: pd.DataFrame, y: np.ndarray, arm: np.ndarray, seed: int = 0):
        self.models = {a: LGBMRegressor(random_state=seed, **PARAMS).fit(x[arm == a], y[arm == a]) for a in ARMS}
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({a: m.predict(x) for a, m in self.models.items()}, index=x.index)


def model_policy(pred_value: pd.DataFrame, margin: float, cost: float, spend_scale: float = 1.0,
                 top_share: float | None = None) -> np.ndarray:
    """Send the arm with the highest predicted profit; no e-mail if no e-mail is predicted to pay for itself.
    `spend_scale` converts a proxy prediction (e.g. visit probability) into dollars; `top_share` keeps only the
    customers with the largest predicted gain (a budget cap)."""
    gain = pd.DataFrame({a: margin * spend_scale * (pred_value[a] - pred_value["control"]) - cost
                         for a in ("mens", "womens")})
    best = gain.idxmax(axis=1).to_numpy()
    best_gain = gain.max(axis=1).to_numpy()
    assign = np.where(best_gain > 0, best, "control")
    if top_share is not None:
        keep = np.zeros(len(assign), dtype=bool)
        keep[np.argsort(-best_gain)[: int(len(assign) * top_share)]] = True
        assign = np.where(keep & (best_gain > 0), best, "control")
    return assign


def rule_policy(df: pd.DataFrame) -> np.ndarray:
    """A sensible hand-written rule: e-mail the category the customer bought last year (women's if both)."""
    return np.where(df["bought_womens"] == 1, "womens", np.where(df["bought_mens"] == 1, "mens", "control"))


def paired_bootstrap(policies: dict[str, np.ndarray], arm: np.ndarray, spend: np.ndarray, margin: float, cost: float,
                     shares: dict[str, float], reps: int = 500, seed: int = 0, baseline: str = "no e-mail") -> pd.DataFrame:
    """Value of every policy and its gain over the baseline, with 95% CIs from the same resampled customers."""
    rng = np.random.default_rng(seed)
    n = len(arm)
    point = {k: value(v, arm, spend, margin, cost, shares) for k, v in policies.items()}
    boots = {k: [] for k in policies}
    for _ in range(reps):
        i = rng.integers(0, n, n)
        for k, v in policies.items():
            boots[k].append(value(v[i], arm[i], spend[i], margin, cost, shares))
    base = np.array(boots[baseline])
    rows = []
    for k in policies:
        d = np.array(boots[k]) - base
        rows.append({"policy": k, "profit_per_customer": point[k], "gain_vs_baseline": point[k] - point[baseline],
                     "gain_lo": float(np.percentile(d, 2.5)), "gain_hi": float(np.percentile(d, 97.5)),
                     "emailed_share": float((policies[k] != "control").mean())})
    return pd.DataFrame(rows)

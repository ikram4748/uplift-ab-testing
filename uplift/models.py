"""Uplift models: who changes their behaviour *because* of the e-mail?

The average effect hides that most customers would buy (or not) either way. An uplift model predicts the
*individual* effect tau(x) = E[Y | x, e-mail] - E[Y | x, no e-mail], so the e-mail can go only to people it moves.

    SLearner            one model with the treatment as a feature; tau = f(x, 1) - f(x, 0)
    TLearner            one model per arm; tau = f1(x) - f0(x)
    XLearner            T-learner, then models of the imputed individual effects (Kunzel et al., 2019)
    TransformedOutcome  regress z = y (t - e) / (e (1 - e)) on x; E[z | x] = tau(x) when e is the true
                        propensity, which here is known (randomized 50/50 between the two arms compared)

All use LightGBM with deliberately conservative settings (shallow trees, large leaves): the signal is weak and
the outcome noisy, so the risk is overfitting, not underfitting. Settings are fixed in advance, not tuned on the
test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def _model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(random_state=seed, **PARAMS)


class SLearner:
    name = "S-learner"

    def fit(self, x, y, t, seed=0):
        self.m = _model(seed).fit(x.assign(_t=t), y)
        return self

    def predict(self, x):
        return self.m.predict(x.assign(_t=1)) - self.m.predict(x.assign(_t=0))


class TLearner:
    name = "T-learner"

    def fit(self, x, y, t, seed=0):
        self.m1 = _model(seed).fit(x[t == 1], y[t == 1])
        self.m0 = _model(seed).fit(x[t == 0], y[t == 0])
        return self

    def predict(self, x):
        return self.m1.predict(x) - self.m0.predict(x)


class XLearner:
    name = "X-learner"

    def fit(self, x, y, t, seed=0):
        base = TLearner().fit(x, y, t, seed)
        d1 = y[t == 1] - base.m0.predict(x[t == 1])        # treated: observed minus predicted untreated
        d0 = base.m1.predict(x[t == 0]) - y[t == 0]        # control: predicted treated minus observed
        self.g1 = _model(seed).fit(x[t == 1], d1)
        self.g0 = _model(seed).fit(x[t == 0], d0)
        self.e = float(t.mean())                           # propensity: known, constant (randomized)
        return self

    def predict(self, x):
        return self.e * self.g0.predict(x) + (1 - self.e) * self.g1.predict(x)


class TransformedOutcome:
    name = "Transformed outcome"

    def fit(self, x, y, t, seed=0):
        e = float(t.mean())
        z = y * (t - e) / (e * (1 - e))
        self.m = _model(seed).fit(x, z)
        return self

    def predict(self, x):
        return self.m.predict(x)


LEARNERS = [SLearner, TLearner, XLearner, TransformedOutcome]


# ---------------------------------------------------------------------------------------------------------------
# Evaluation: there is no ground truth for an individual effect, so rank quality is measured on held-out data
# ---------------------------------------------------------------------------------------------------------------

def qini_curve(score: np.ndarray, y: np.ndarray, t: np.ndarray, bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Target customers in order of predicted uplift. At each fraction phi, the incremental outcome is
    Y_t(phi) - Y_c(phi) * N_t(phi) / N_c(phi): what the treated in the top phi gained over comparable controls.
    Returns (fractions, incremental outcome per customer in the whole population)."""
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    cy_t, cy_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    cn_t, cn_c = np.cumsum(t), np.cumsum(1 - t)
    idx = np.unique(np.linspace(0, len(y) - 1, bins + 1).astype(int))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(cn_c[idx] > 0, cy_t[idx] - cy_c[idx] * cn_t[idx] / cn_c[idx], 0.0)
    frac = (idx + 1) / len(y)
    return np.concatenate([[0], frac]), np.concatenate([[0], gain / len(y)])


def qini_coefficient(score: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """Area between the model's Qini curve and the random-targeting line (0 = no better than random)."""
    frac, gain = qini_curve(score, y, t)
    random_line = frac * gain[-1]
    return float(np.trapezoid(gain - random_line, frac))


def uplift_at(score: np.ndarray, y: np.ndarray, t: np.ndarray, share: float) -> float:
    """Average treatment effect among the top `share` of customers by predicted uplift."""
    top = np.argsort(-score)[: int(len(score) * share)]
    yt, tt = y[top], t[top]
    return float(yt[tt == 1].mean() - yt[tt == 0].mean())


def fit_predict(learner_cls, x_train: pd.DataFrame, y_train, t_train, x_test: pd.DataFrame, seed: int = 0):
    return learner_cls().fit(x_train, np.asarray(y_train, float), np.asarray(t_train, int), seed).predict(x_test)

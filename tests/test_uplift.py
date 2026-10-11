"""Every estimator is checked on synthetic data where the true answer is known."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift import experiment as ex, models, policy, power

rng = np.random.default_rng(42)


def test_srm_flags_a_broken_split():
    assert ex.srm({"a": 21_306, "b": 21_307, "c": 21_387})[1] > 0.5
    assert ex.srm({"a": 20_000, "b": 21_000})[1] < 1e-5


def test_diff_in_means_recovers_the_effect():
    y_c = rng.normal(10, 2, 50_000)
    y_t = rng.normal(10.5, 2, 50_000)
    e = ex.diff_in_means(y_t, y_c)
    assert e.lo < 0.5 < e.hi and e.p < 1e-10 and e.relative == pytest.approx(0.05, abs=0.005)


@pytest.mark.parametrize("rho", [0.3, 0.7])
def test_cuped_removes_rho_squared_of_the_variance_and_stays_unbiased(rho):
    n = 100_000
    pre = rng.normal(size=n)
    t = rng.random(n) < 0.5
    y = rho * pre + np.sqrt(1 - rho ** 2) * rng.normal(size=n) + 0.1 * t
    est, theta, reduction = ex.cuped(y, pre, t)
    assert reduction == pytest.approx(rho ** 2, abs=0.02)
    assert theta == pytest.approx(rho, abs=0.02)
    assert est.lo < 0.1 < est.hi
    assert est.se < ex.diff_in_means(y[t], y[~t]).se


def test_regression_adjustment_is_unbiased():
    n = 40_000
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    t = rng.random(n) < 0.5
    y = 2 * x["a"].to_numpy() - x["b"].to_numpy() + 0.3 * t + rng.normal(size=n)
    est = ex.regression_adjusted(y, t, x)
    assert est.lo < 0.3 < est.hi and est.se < ex.diff_in_means(y[t], y[~t]).se / 2


def test_holm_matches_hand_calculation():
    assert ex.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_balance_is_near_zero_under_randomization():
    df = pd.DataFrame({"x": rng.normal(size=30_000), "c": rng.choice(["u", "v"], 30_000),
                       "arm": rng.choice(["control", "t"], 30_000)})
    assert ex.balance(df, ["x", "c"]).abs().max().max() < 0.05


def test_sample_size_formula():
    assert power.n_per_arm_proportion(0.10, 0.01) == pytest.approx(14_748, abs=5)
    n = 14_748
    assert power.mde_proportion(0.10, n) == pytest.approx(0.01, abs=2e-4)


def test_peeking_inflates_false_positives():
    s = power.peeking_simulation(p=0.1, n_per_arm=5_000, looks=10, sims=1500, seed=1)
    assert s["fixed_horizon_fpr"] < 0.08 and s["peeking_fpr"] > 2 * s["fixed_horizon_fpr"]


def synthetic_uplift(n=40_000):
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    tau = np.where(x["a"] > 0, 0.4, 0.0)                     # only half the population responds
    t = (rng.random(n) < 0.5).astype(int)
    y = 0.5 * x["b"].to_numpy() + tau * t + rng.normal(size=n)
    return x, y, t, tau


@pytest.mark.parametrize("learner", models.LEARNERS, ids=lambda c: c.name)
def test_learners_find_who_responds(learner):
    x, y, t, tau = synthetic_uplift()
    pred = models.fit_predict(learner, x[:30_000], y[:30_000], t[:30_000], x[30_000:])
    assert np.corrcoef(pred, tau[30_000:])[0, 1] > 0.5


def test_qini_rewards_good_rankings():
    x, y, t, tau = synthetic_uplift()
    oracle = models.qini_coefficient(tau + 1e-6 * rng.random(len(tau)), y, t)
    random = models.qini_coefficient(rng.random(len(tau)), y, t)
    assert oracle > 0.02 and abs(random) < oracle / 3
    assert models.uplift_at(tau + 1e-6 * rng.random(len(tau)), y, t, 0.3) == pytest.approx(0.4, abs=0.06)


def test_ipw_policy_value_matches_the_truth():
    n = 150_000
    arm = rng.choice(policy.ARMS, n)
    effect = {"control": 0.0, "mens": 2.0, "womens": 1.0}
    spend = rng.exponential(1.0, n) + np.array([effect[a] for a in arm])
    shares = {a: 1 / 3 for a in policy.ARMS}
    assign = np.full(n, "mens")
    v = policy.value(assign, arm, spend, margin=0.5, cost=0.1, shares=shares)
    assert v == pytest.approx(0.5 * (1.0 + 2.0) - 0.1, abs=0.03)          # true value under "everyone gets mens"


def test_model_policy_skips_unprofitable_customers():
    pred = pd.DataFrame({"control": [1.0, 1.0], "mens": [3.0, 1.05], "womens": [2.0, 1.0]})
    assert policy.model_policy(pred, margin=0.5, cost=0.1).tolist() == ["mens", "control"]

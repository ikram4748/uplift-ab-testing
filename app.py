"""Streamlit dashboard for the experiment readout, uplift models and targeting decision.

    python -m analysis.run_all     # once, to produce results/
    streamlit run app.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from uplift import experiment as ex, policy, power
from uplift.data import ROOT

st.set_page_config(page_title="E-mail Experiment & Uplift", page_icon="📈", layout="wide")
RES = ROOT / "results"
COLORS = {"mens": "#2f6db3", "womens": "#c2477a", "control": "#8a8f98"}


@st.cache_data
def results() -> dict | None:
    p = RES / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def predictions() -> tuple[pd.DataFrame, dict]:
    return pd.read_csv(RES / "policy_predictions.csv.gz"), json.loads((RES / "policy_meta.json").read_text())


def tab_readout(res: dict) -> None:
    r = res["readout"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{r['n']:,}")
    c2.metric("Sample-ratio check (p)", f"{r['srm']['p']:.2f}", help="Chi-square test of arm sizes against a 1/3 split. "
              "A tiny p-value would mean the randomization or logging is broken.")
    c3.metric("Largest covariate imbalance", f"{r['balance_max_abs_smd']:.3f}", help="Max |standardized mean difference| "
              "of pre-treatment covariates between arms. Below 0.1 = balanced.")
    eff = pd.DataFrame(r["effects"])
    metric = st.radio("Metric", ["visit", "conversion", "spend"], horizontal=True)
    e = eff[eff.metric == metric]
    fig = go.Figure()
    for _, row in e.iterrows():
        fig.add_trace(go.Scatter(x=[row.effect], y=[f"{row.arm} e-mail"], mode="markers", marker=dict(size=12, color=COLORS[row.arm]),
                                 error_x=dict(type="data", symmetric=False, array=[row.hi - row.effect],
                                              arrayminus=[row.effect - row.lo]), showlegend=False))
    fig.add_vline(x=0, line_color="#999")
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), title=f"Effect on {metric} vs no e-mail (95% CI)",
                      xaxis_tickformat="$.2f" if metric == "spend" else ".1%")
    st.plotly_chart(fig, width="stretch")
    show = e[["arm", "control_mean", "treated_mean", "effect", "lo", "hi", "relative", "p_holm"]].rename(
        columns={"control_mean": "no e-mail", "treated_mean": "e-mail", "p_holm": "p (Holm)"})
    st.dataframe(show, hide_index=True, width="stretch")


def tab_cuped(res: dict) -> None:
    eff = pd.DataFrame(res["readout"]["effects"])
    st.markdown(f"On this data CUPED with last year's spend cut variance by at most **{eff.cuped_variance_reduction.max():.1%}**: "
                "the covariate barely predicts a two-week outcome. CUPED removes about **rho squared** of the variance, "
                "where rho is the correlation between the pre-period covariate and the outcome. Try it:")
    rho = st.slider("Correlation between the pre-period metric and the outcome (rho)", 0.0, 0.95, 0.7, 0.05)
    n = st.select_slider("Customers per arm", [2_000, 5_000, 10_000, 20_000, 50_000], 10_000)
    rng = np.random.default_rng(0)
    pre = rng.normal(size=2 * n)
    t = np.arange(2 * n) < n
    y = rho * pre + np.sqrt(1 - rho ** 2) * rng.normal(size=2 * n) + 0.03 * t
    plain = ex.diff_in_means(y[t], y[~t])
    cu, _, red = ex.cuped(y, pre, t)
    c1, c2, c3 = st.columns(3)
    c1.metric("Variance reduction", f"{red:.0%}", f"theory {rho ** 2:.0%}", delta_color="off")
    c2.metric("95% CI width, plain", f"{2 * 1.96 * plain.se:.4f}")
    c3.metric("95% CI width, CUPED", f"{2 * 1.96 * cu.se:.4f}", f"{cu.se / plain.se - 1:.0%}")
    st.caption(f"Equivalent to running the plain test with {1 / (1 - red):.1f}x the customers.")


def tab_uplift(res: dict) -> None:
    c1, c2 = st.columns(2)
    arm = c1.radio("E-mail", ["mens", "womens"], horizontal=True)
    y = c2.radio("Outcome", ["visit", "spend"], horizontal=True)
    scale, label = (1000, "incremental $ per 1,000 customers") if y == "spend" else (100, "incremental visits per 100 customers")
    fig = go.Figure()
    for key, (frac, gain) in res["uplift"]["curves"].items():
        a, o, name = key.split("|")
        if a == arm and o == y:
            fig.add_trace(go.Scatter(x=frac, y=np.array(gain) * scale, name=name, mode="lines",
                                     line=dict(dash="dot" if name == "random" else "solid")))
    fig.update_layout(height=380, xaxis_title="share of customers e-mailed, best-ranked first", yaxis_title=label,
                      title="Qini curve (held-out split 0): higher above the random line = better targeting")
    st.plotly_chart(fig, width="stretch")
    u = pd.DataFrame(res["uplift"]["rows"])
    u = u[(u.arm == arm) & (u.outcome == y)]
    agg = u.groupby("model").agg(qini=("qini", "mean"), qini_sd=("qini", "std"), effect_top30=("uplift_top30", "mean"),
                                 effect_all=("ate", "mean")).sort_values("qini", ascending=False)
    agg[["qini", "qini_sd"]] *= scale
    st.dataframe(agg, width="stretch")
    st.caption("Mean and standard deviation over 5 random train/test splits. A standard deviation as large as the mean "
               "means the ranking is not reliably better than random.")


def tab_targeting() -> None:
    try:
        df, meta = predictions()
    except FileNotFoundError:
        st.info("Run `python -m analysis.run_all` first.")
        return
    c1, c2 = st.columns(2)
    margin = c1.slider("Gross margin on spend", 0.1, 0.8, meta["margin"], 0.05)
    cost = c2.slider("Cost per e-mail ($)", 0.0, 0.5, 0.15, 0.01, help="Sending, plus fatigue/unsubscribe risk or an incentive.")
    arm, spend = df["arm"].to_numpy(), df["spend"].to_numpy(float)
    p_spend = df[["spend_control", "spend_mens", "spend_womens"]].set_axis(policy.ARMS, axis=1)
    p_visit = df[["visit_control", "visit_mens", "visit_womens"]].set_axis(policy.ARMS, axis=1)
    rng = np.random.default_rng(0)
    pols = {"no e-mail": np.full(len(df), "control"), "everyone: mens": np.full(len(df), "mens"),
            "everyone: womens": np.full(len(df), "womens"), "random arm": rng.choice(policy.ARMS, len(df)),
            "rule: category bought": policy.rule_policy(df), "model: spend": policy.model_policy(p_spend, margin, cost),
            "model: visit proxy": policy.model_policy(p_visit, margin, cost, spend_scale=meta["dollars_per_visit"])}
    base = policy.value(pols["no e-mail"], arm, spend, margin, cost, meta["shares"])
    rows = [{"policy": k, "profit vs no e-mail ($ per 1,000)": 1000 * (policy.value(v, arm, spend, margin, cost, meta["shares"]) - base),
             "share e-mailed": (v != "control").mean()} for k, v in pols.items() if k != "no e-mail"]
    t = pd.DataFrame(rows)
    fig = px.bar(t, x="policy", y="profit vs no e-mail ($ per 1,000)", color="share e-mailed", color_continuous_scale="Blues",
                 range_color=(0, 1))
    fig.add_hline(y=0, line_color="#999")
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")
    best = t.loc[t["profit vs no e-mail ($ per 1,000)"].idxmax()]
    if best["profit vs no e-mail ($ per 1,000)"] <= 0:
        st.warning("At this cost no policy beats sending nothing.")
    else:
        st.success(f"Best: **{best.policy}**, about ${best['profit vs no e-mail ($ per 1,000)']:,.0f} per 1,000 customers, "
                   f"e-mailing {best['share e-mailed']:.0%} of them.")
    st.caption("Estimated by inverse-propensity weighting on held-out customers (split 0, no bootstrap here; see the README "
               "for confidence intervals). Differences of a few dollars per 1,000 are within noise.")


def tab_planning() -> None:
    st.subheader("Sample size")
    c1, c2, c3 = st.columns(3)
    p0 = c1.number_input("Baseline rate", 0.001, 0.9, 0.106, 0.001, format="%.3f")
    lift = c2.number_input("Smallest lift worth detecting (relative)", 0.01, 2.0, 0.10, 0.01)
    pw = c3.select_slider("Power", [0.7, 0.8, 0.9], 0.8)
    st.metric("Customers needed per arm", f"{power.n_per_arm_proportion(p0, p0 * lift, power=pw):,}")
    st.subheader("Why you must not stop early")
    looks = st.slider("Number of looks at the data", 1, 20, 10)
    s = power.peeking_simulation(p=p0, n_per_arm=10_000, looks=looks, sims=1000)
    fig = px.line(x=list(range(1, looks + 1)), y=np.array(s["fpr_by_looks"]) * 100, markers=True,
                  labels={"x": "looks", "y": "false positive rate (%)"})
    fig.add_hline(y=5, line_dash="dash", line_color="#999")
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"1,000 simulated A/A tests (no real effect). Stopping at the first p < 0.05 declares a winner in "
               f"{s['peeking_fpr']:.0%} of them.")


def main() -> None:
    st.title("📈 E-mail Experiment & Uplift Modeling")
    st.caption("Hillstrom MineThatData experiment: 64,000 customers randomly sent a men's e-mail, a women's e-mail, or nothing.")
    res = results()
    if res is None:
        st.info("Run `python -m analysis.run_all` first (about 4 minutes).")
        return
    tabs = st.tabs(["1 · Readout", "2 · CUPED", "3 · Uplift models", "4 · Who to e-mail", "5 · Planning"])
    with tabs[0]:
        tab_readout(res)
    with tabs[1]:
        tab_cuped(res)
    with tabs[2]:
        tab_uplift(res)
    with tabs[3]:
        tab_targeting()
    with tabs[4]:
        tab_planning()


main()

"""Produce every number and figure in the README.

    python -m analysis.run_all            # ~3 minutes on a laptop CPU

Writes results/results.json, results/RESULTS.md, figures/*.png and results/policy_predictions.csv.gz (used by
the dashboard's what-if sliders).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uplift import data, experiment as ex, models, policy, power  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
SEEDS = [0, 1, 2, 3, 4]
MARGIN = 0.30                     # gross margin on spend (assumption; the dashboard lets you change it)
COSTS = [0.05, 0.15, 0.30]        # cost per e-mail in USD: cheap send, send + fatigue, send + incentive
COLORS = {"control": "#8a8f98", "mens": "#2f6db3", "womens": "#c2477a"}


def readout(df: pd.DataFrame) -> dict:
    out = {"n": int(len(df)), "arm_sizes": df["arm"].value_counts().to_dict()}
    chi2, p = ex.srm(out["arm_sizes"])
    out["srm"] = {"chi2": chi2, "p": p}
    smd = ex.balance(df, data.PRE_TREATMENT)
    out["balance_max_abs_smd"] = float(smd.abs().max().max())
    x_all = data.features(df)
    rows = []
    for arm in data.TREATMENTS:
        d = df[df["arm"].isin([arm, "control"])]
        t = (d["arm"] == arm).to_numpy()
        for y in data.OUTCOMES:
            v = d[y].to_numpy(float)
            dm = ex.diff_in_means(v[t], v[~t])
            cu, theta, red = ex.cuped(v, d["history"].to_numpy(float), t)
            ra = ex.regression_adjusted(v, t, x_all.loc[d.index])
            row = {"arm": arm, "metric": y, "control_mean": dm.control_mean, "treated_mean": float(v[t].mean()),
                   "effect": dm.effect, "lo": dm.lo, "hi": dm.hi, "relative": dm.relative, "p": dm.p,
                   "cuped_variance_reduction": red, "cuped_se": cu.se,
                   "regadj_effect": ra.effect, "regadj_se": ra.se, "regadj_variance_reduction": 1 - (ra.se / dm.se) ** 2}
            if y == "spend":
                row["bootstrap_lo"], row["bootstrap_hi"] = ex.bootstrap_ci(v[t], v[~t])
            rows.append(row)
    for r, adj in zip(rows, ex.holm([r["p"] for r in rows])):
        r["p_holm"] = adj
    out["effects"] = rows
    return out


def cuped_simulation(n: int = 20_000, effect: float = 0.02, seed: int = 0) -> list[dict]:
    """Where CUPED pays off: a pre-period metric correlated rho with the outcome cuts variance by about rho^2."""
    rng = np.random.default_rng(seed)
    rows = []
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
        pre = rng.normal(size=2 * n)
        y = rho * pre + np.sqrt(1 - rho ** 2) * rng.normal(size=2 * n)
        t = np.arange(2 * n) < n
        y = y + effect * t
        dm = ex.diff_in_means(y[t], y[~t])
        cu, _, red = ex.cuped(y, pre, t)
        rows.append({"rho": rho, "variance_reduction": red, "theory": rho ** 2,
                     "ci_width_plain": 2 * 1.96 * dm.se, "ci_width_cuped": 2 * 1.96 * cu.se,
                     "sample_size_needed_vs_plain": 1 - red})
    return rows


def planning(df: pd.DataFrame) -> dict:
    c = df[df["arm"] == "control"]
    n = int((df["arm"] == "control").sum())
    out = {"n_per_arm": n}
    for y in ("visit", "conversion"):
        p0 = float(c[y].mean())
        out[y] = {"baseline": p0, "mde_abs": power.mde_proportion(p0, n), "n_for_10pct_relative_lift":
                  power.n_per_arm_proportion(p0, 0.1 * p0)}
    out["peeking"] = power.peeking_simulation(p=float(c["visit"].mean()), n_per_arm=20_000, looks=10, sims=2000)
    return out


def uplift_models(df: pd.DataFrame, x: pd.DataFrame) -> dict:
    rows, curves = [], {}
    for arm in data.TREATMENTS:
        d = df[df["arm"].isin([arm, "control"])]
        xd = x.loc[d.index]
        rule = d["bought_mens" if arm == "mens" else "bought_womens"].to_numpy(float)
        for y in ("visit", "spend"):
            yv, tv = d[y].to_numpy(float), (d["arm"] == arm).to_numpy(int)
            for seed in SEEDS:
                tr, te = data.split(d, seed)
                rng = np.random.default_rng(seed)
                scores = {"random": rng.random(len(te)), "rule: bought this category": rule[te] + 1e-6 * rng.random(len(te))}
                for cls in models.LEARNERS:
                    scores[cls.name] = models.fit_predict(cls, xd.iloc[tr], yv[tr], tv[tr], xd.iloc[te], seed)
                ate = float(yv[te][tv[te] == 1].mean() - yv[te][tv[te] == 0].mean())
                for name, s in scores.items():
                    rows.append({"arm": arm, "outcome": y, "seed": seed, "model": name,
                                 "qini": models.qini_coefficient(s, yv[te], tv[te]),
                                 "uplift_top30": models.uplift_at(s, yv[te], tv[te], 0.3), "ate": ate})
                    if seed == 0:
                        curves[f"{arm}|{y}|{name}"] = [a.tolist() for a in models.qini_curve(s, yv[te], tv[te], bins=50)]
    return {"rows": rows, "curves": curves}


def targeting(df: pd.DataFrame, x: pd.DataFrame) -> dict:
    arm = df["arm"].to_numpy()
    shares = (df["arm"].value_counts() / len(df)).to_dict()
    tables, saved = [], None
    for seed in SEEDS:
        tr, te = data.split(df, seed)
        spend_m = policy.ArmModels().fit(x.iloc[tr], df["spend"].to_numpy(float)[tr], arm[tr], seed)
        visit_m = policy.ArmModels().fit(x.iloc[tr], df["visit"].to_numpy(float)[tr], arm[tr], seed)
        p_spend, p_visit = spend_m.predict(x.iloc[te]), visit_m.predict(x.iloc[te])
        # converts a predicted visit lift into dollars: incremental spend per incremental visit (training data only)
        trd = df.iloc[tr]
        lift = lambda col: trd[trd.arm != "control"][col].mean() - trd[trd.arm == "control"][col].mean()  # noqa: E731
        dollars_per_visit = float(lift("spend") / lift("visit"))
        rng = np.random.default_rng(seed)
        for cost in COSTS:
            pols = {"no e-mail": np.full(len(te), "control"), "everyone: mens": np.full(len(te), "mens"),
                    "everyone: womens": np.full(len(te), "womens"),
                    "random arm": rng.choice(policy.ARMS, len(te)), "rule: category bought": policy.rule_policy(df.iloc[te]),
                    "model: spend": policy.model_policy(p_spend, MARGIN, cost),
                    "model: visit proxy": policy.model_policy(p_visit, MARGIN, cost, spend_scale=dollars_per_visit)}
            t = policy.paired_bootstrap(pols, arm[te], df["spend"].to_numpy(float)[te], MARGIN, cost, shares,
                                        reps=300, seed=seed)
            t["cost"], t["seed"] = cost, seed
            tables.append(t)
        if seed == 0:
            saved = pd.concat([df.iloc[te][["arm", "spend", "visit", "bought_mens", "bought_womens"]].reset_index(drop=True),
                               p_spend.add_prefix("spend_").reset_index(drop=True),
                               p_visit.add_prefix("visit_").reset_index(drop=True)], axis=1)
            saved.attrs["dollars_per_visit"] = dollars_per_visit
    all_t = pd.concat(tables)
    saved.to_csv(RESULTS / "policy_predictions.csv.gz", index=False)
    (RESULTS / "policy_meta.json").write_text(json.dumps({"dollars_per_visit": saved.attrs["dollars_per_visit"],
                                                           "shares": shares, "margin": MARGIN}))
    return {"table": all_t.to_dict(orient="records"), "shares": shares}


# ---------------------------------------------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------------------------------------------

def figures(res: dict) -> None:
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

    eff = pd.DataFrame(res["readout"]["effects"])
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.2), sharey=True)
    for ax, m in zip(axes, data.OUTCOMES):
        e = eff[eff.metric == m].reset_index(drop=True)
        for i, r in e.iterrows():
            ax.errorbar(r.effect, i, xerr=[[r.effect - r.lo], [r.hi - r.effect]], fmt="o", color=COLORS[r.arm], capsize=3)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_ylim(-0.6, len(e) - 0.4)
        ax.set_xlim(left=min(0, e.lo.min()) - 0.05 * e.hi.max(), right=e.hi.max() * 1.1)
        ax.xaxis.set_major_formatter(FuncFormatter(
            (lambda v, _: f"${v:.2f}") if m == "spend" else (lambda v, _: f"{v * 100:.1f} pt")))
        ax.set_title({"visit": "Visit rate", "conversion": "Conversion rate", "spend": "Spend per customer"}[m])
    axes[0].set_yticks(range(len(e)), [f"{a} e-mail" for a in e.arm])
    fig.suptitle("Average effect vs no e-mail, 95% CI", y=1.1)
    fig.savefig(FIGURES / "effects.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, y in zip(axes, ("visit", "spend")):
        for name, color in (("random", "#999"), ("rule: bought this category", "#e0a030"), ("T-learner", "#2f6db3"),
                            ("X-learner", "#3a9a5b"), ("Transformed outcome", "#c2477a"), ("S-learner", "#7b5bb5")):
            frac, gain = res["uplift"]["curves"][f"mens|{y}|{name}"]
            ax.plot(frac, np.array(gain) * (1000 if y == "spend" else 100), label=name, color=color, lw=1.4)
        ax.set_xlabel("share of customers e-mailed (ranked by model)")
        ax.set_ylabel("incremental visits per 100" if y == "visit" else "incremental $ per 1,000")
        ax.set_title(f"Qini curve: men's e-mail, outcome = {y}")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(FIGURES / "qini.png", bbox_inches="tight")
    plt.close(fig)

    pk = res["planning"]["peeking"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot(range(1, pk["looks"] + 1), np.array(pk["fpr_by_looks"]) * 100, marker="o", color="#c2477a")
    ax.axhline(5, color="#999", ls="--", lw=0.8)
    ax.set_xlabel("number of looks at the data")
    ax.set_ylabel("false positive rate (%)")
    ax.set_title("A/A tests: stopping at the first p < 0.05")
    fig.savefig(FIGURES / "peeking.png", bbox_inches="tight")
    plt.close(fig)

    t = pd.DataFrame(res["targeting"]["table"])
    g = t.groupby(["cost", "policy"])["gain_vs_baseline"].mean().unstack(0) * 1000
    order = ["everyone: mens", "everyone: womens", "random arm", "rule: category bought", "model: spend", "model: visit proxy"]
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.26
    for j, cost in enumerate(COSTS):
        ax.bar(np.arange(len(order)) + (j - 1) * w, g.loc[order, cost], w, label=f"${cost:.2f} per e-mail")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(range(len(order)), order, rotation=20, ha="right")
    ax.set_ylabel("profit vs no e-mail ($ per 1,000)")
    ax.set_title(f"Targeting policies on held-out customers (margin {MARGIN:.0%}, mean of {len(SEEDS)} splits)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIGURES / "policies.png", bbox_inches="tight")
    plt.close(fig)


def report(res: dict) -> str:
    r, pl = res["readout"], res["planning"]
    eff = pd.DataFrame(r["effects"])
    L = ["# Results", "", f"Hillstrom e-mail experiment: {r['n']:,} customers, three arms. Generated {time.strftime('%Y-%m-%d')} "
         f"by `python -m analysis.run_all`.", "", "## 1. Is the experiment valid?", "",
         f"- **Sample-ratio mismatch:** arm sizes {', '.join(f'{k} {v:,}' for k, v in r['arm_sizes'].items())}; "
         f"chi-square p = {r['srm']['p']:.2f} (no mismatch).",
         f"- **Covariate balance:** largest |standardized mean difference| across all pre-treatment covariates = "
         f"{r['balance_max_abs_smd']:.3f} (< 0.1, balanced).", "", "## 2. Average effects", "",
         "| E-mail | Metric | No e-mail | E-mail | Effect [95% CI] | Relative | Holm p |", "|---|---|---|---|---|---|---|"]
    for _, e in eff.iterrows():
        f = (lambda v: f"${v:.3f}") if e.metric == "spend" else (lambda v: f"{v:.2%}")
        ci = f"{f(e.effect)} [{f(e.lo)}, {f(e.hi)}]"
        L.append(f"| {e.arm} | {e.metric} | {f(e.control_mean)} | {f(e.treated_mean)} | {ci} | {e.relative:+.0%} | {e.p_holm:.1e} |")
    sp = eff[eff.metric == "spend"]
    L += ["", "Spend is heavy-tailed (most customers spend $0), so its CI was also checked with a percentile bootstrap: "
          + "; ".join(f"{e.arm} [${e.bootstrap_lo:.3f}, ${e.bootstrap_hi:.3f}]" for _, e in sp.iterrows()) + ".", "",
          "## 3. Variance reduction (CUPED)", "",
          f"CUPED with last year's spend (`history`) as the pre-period covariate reduced variance by at most "
          f"**{eff.cuped_variance_reduction.max():.1%}**, and regression adjustment with all pre-treatment covariates by at most "
          f"**{eff.regadj_variance_reduction.max():.1%}**. The reason: CUPED removes the share of variance a covariate explains "
          "(about rho squared), and last year's total spend barely predicts a two-week visit or purchase. A simulation with a "
          "pre-period version of the same metric shows when it pays off:", "",
          "| Correlation rho | Variance reduction | Theory (rho squared) | Sample size needed vs plain |", "|---|---|---|---|"]
    for s in res["cuped_simulation"]:
        L.append(f"| {s['rho']:.1f} | {s['variance_reduction']:.0%} | {s['theory']:.0%} | {s['sample_size_needed_vs_plain']:.0%} |")
    L += ["", "## 4. Planning and peeking", "",
          f"With {pl['n_per_arm']:,} customers per arm, the smallest detectable lift (80% power, alpha 0.05) is "
          f"**{pl['visit']['mde_abs']:.2%}** absolute on the {pl['visit']['baseline']:.1%} visit rate and "
          f"**{pl['conversion']['mde_abs']:.2%}** on the {pl['conversion']['baseline']:.2%} conversion rate. A 10% relative lift "
          f"in conversion would need {pl['conversion']['n_for_10pct_relative_lift']:,} customers per arm.", "",
          f"Peeking: in {pl['peeking']['sims']:,} simulated A/A tests checked {pl['peeking']['looks']} times, stopping at the first "
          f"p < 0.05 gives a **{pl['peeking']['peeking_fpr']:.0%} false-positive rate** instead of "
          f"{pl['peeking']['fixed_horizon_fpr']:.0%} at a fixed horizon.", "", "## 5. Uplift models (held-out 30%, 5 random splits)", "",
          "Qini coefficient = area between the model's Qini curve and random targeting (higher = better ranking of who "
          "responds; 0 = no better than random). Uplift in top 30% = average effect among the 30% the model ranks highest, "
          "compared with the average effect for everyone.", "",
          "| E-mail | Outcome | Model | Qini (mean ± sd) | Effect in top 30% | Effect for everyone |", "|---|---|---|---|---|---|"]
    u = pd.DataFrame(res["uplift"]["rows"])
    for (arm, y), g in u.groupby(["arm", "outcome"], sort=False):
        scale, unit = (1000, " ($ per 1,000)") if y == "spend" else (100, " (per 100)")
        agg = g.groupby("model").agg(q=("qini", "mean"), qs=("qini", "std"), top=("uplift_top30", "mean"), ate=("ate", "mean"))
        for name, a in agg.sort_values("q", ascending=False).iterrows():
            fmt = (lambda v: f"${v:.3f}") if y == "spend" else (lambda v: f"{v:.2%}")
            L.append(f"| {arm} | {y} | {name} | {a.q * scale:.2f} ± {a.qs * scale:.2f}{unit} | {fmt(a.top)} | {fmt(a.ate)} |")
    t = pd.DataFrame(res["targeting"]["table"])
    L += ["", f"## 6. Who should get which e-mail? (margin {MARGIN:.0%}; profit vs sending nothing, $ per 1,000 customers)", "",
          "Mean over 5 held-out splits; the 95% CI is the paired bootstrap on split 0.", "",
          "| Policy | " + " | ".join(f"${c:.2f} / e-mail" for c in COSTS) + " | Share e-mailed ($0.15) |",
          "|---|" + "---|" * (len(COSTS) + 1)]
    for pol in ["everyone: mens", "everyone: womens", "random arm", "rule: category bought", "model: spend", "model: visit proxy"]:
        cells = []
        for c in COSTS:
            s = t[(t.policy == pol) & (t.cost == c)]
            s0 = s[s.seed == 0].iloc[0]
            cells.append(f"{s.gain_vs_baseline.mean() * 1000:+.0f} [{s0.gain_lo * 1000:+.0f}, {s0.gain_hi * 1000:+.0f}]")
        share = t[(t.policy == pol) & (t.cost == 0.15)].emailed_share.mean()
        L.append(f"| {pol} | " + " | ".join(cells) + f" | {share:.0%} |")
    text = "\n".join(L) + "\n"
    (RESULTS / "RESULTS.md").write_text(text)
    return text


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    df = data.load()
    x = data.features(df)
    res = {"readout": readout(df), "cuped_simulation": cuped_simulation(), "planning": planning(df)}
    print("readout done", f"{time.time() - t0:.0f}s")
    res["uplift"] = uplift_models(df, x)
    print("uplift models done", f"{time.time() - t0:.0f}s")
    res["targeting"] = targeting(df, x)
    print("targeting done", f"{time.time() - t0:.0f}s")
    (RESULTS / "results.json").write_text(json.dumps(res, indent=1, default=float))
    figures(res)
    print(report(res))


if __name__ == "__main__":
    main()

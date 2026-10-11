# Results

Hillstrom e-mail experiment: 64,000 customers, three arms. Generated 2026-10-11 by `python -m analysis.run_all`.

## 1. Is the experiment valid?

- **Sample-ratio mismatch:** arm sizes womens 21,387, mens 21,307, control 21,306; chi-square p = 0.90 (no mismatch).
- **Covariate balance:** largest |standardized mean difference| across all pre-treatment covariates = 0.014 (< 0.1, balanced).

## 2. Average effects

| E-mail | Metric | No e-mail | E-mail | Effect [95% CI] | Relative | Holm p |
|---|---|---|---|---|---|---|
| mens | visit | 10.62% | 18.28% | 7.66% [7.00%, 8.32%] | +72% | 1.6e-112 |
| mens | conversion | 0.57% | 1.25% | 0.68% [0.50%, 0.86%] | +119% | 5.9e-13 |
| mens | spend | $0.653 | $1.423 | $0.770 [$0.485, $1.055] | +118% | 3.5e-07 |
| womens | visit | 10.62% | 15.14% | 4.52% [3.89%, 5.16%] | +43% | 9.7e-44 |
| womens | conversion | 0.57% | 0.88% | 0.31% [0.15%, 0.47%] | +54% | 3.1e-04 |
| womens | spend | $0.653 | $1.077 | $0.424 [$0.169, $0.680] | +65% | 1.1e-03 |

Spend is heavy-tailed (most customers spend $0), so its CI was also checked with a percentile bootstrap: mens [$0.491, $1.054]; womens [$0.173, $0.674].

## 3. Variance reduction (CUPED)

CUPED with last year's spend (`history`) as the pre-period covariate reduced variance by at most **0.4%**, and regression adjustment with all pre-treatment covariates by at most **2.9%**. The reason: CUPED removes the share of variance a covariate explains (about rho squared), and last year's total spend barely predicts a two-week visit or purchase. A simulation with a pre-period version of the same metric shows when it pays off:

| Correlation rho | Variance reduction | Theory (rho squared) | Sample size needed vs plain |
|---|---|---|---|
| 0.0 | 0% | 0% | 100% |
| 0.3 | 9% | 9% | 91% |
| 0.5 | 25% | 25% | 75% |
| 0.7 | 49% | 49% | 51% |
| 0.9 | 81% | 81% | 19% |

## 4. Planning and peeking

With 21,306 customers per arm, the smallest detectable lift (80% power, alpha 0.05) is **0.85%** absolute on the 10.6% visit rate and **0.22%** on the 0.57% conversion rate. A 10% relative lift in conversion would need 286,118 customers per arm.

Peeking: in 2,000 simulated A/A tests checked 10 times, stopping at the first p < 0.05 gives a **18% false-positive rate** instead of 4% at a fixed horizon.

## 5. Uplift models (held-out 30%, 5 random splits)

Qini coefficient = area between the model's Qini curve and random targeting (higher = better ranking of who responds; 0 = no better than random). Uplift in top 30% = average effect among the 30% the model ranks highest, compared with the average effect for everyone.

| E-mail | Outcome | Model | Qini (mean ± sd) | Effect in top 30% | Effect for everyone |
|---|---|---|---|---|---|
| mens | visit | S-learner | 0.10 ± 0.13 (per 100) | 9.10% | 7.83% |
| mens | visit | rule: bought this category | 0.07 ± 0.11 (per 100) | 8.27% | 7.83% |
| mens | visit | X-learner | 0.05 ± 0.07 (per 100) | 8.16% | 7.83% |
| mens | visit | T-learner | 0.05 ± 0.08 (per 100) | 8.40% | 7.83% |
| mens | visit | Transformed outcome | 0.03 ± 0.06 (per 100) | 8.14% | 7.83% |
| mens | visit | random | -0.01 ± 0.06 (per 100) | 7.31% | 7.83% |
| mens | spend | Transformed outcome | 23.46 ± 14.14 ($ per 1,000) | $1.032 | $0.819 |
| mens | spend | T-learner | 21.38 ± 6.66 ($ per 1,000) | $0.930 | $0.819 |
| mens | spend | X-learner | 19.08 ± 15.43 ($ per 1,000) | $0.917 | $0.819 |
| mens | spend | rule: bought this category | 11.43 ± 15.72 ($ per 1,000) | $0.891 | $0.819 |
| mens | spend | S-learner | -1.04 ± 44.53 ($ per 1,000) | $0.777 | $0.819 |
| mens | spend | random | -5.52 ± 32.27 ($ per 1,000) | $0.674 | $0.819 |
| womens | visit | rule: bought this category | 0.40 ± 0.07 (per 100) | 7.60% | 4.59% |
| womens | visit | S-learner | 0.35 ± 0.07 (per 100) | 7.24% | 4.59% |
| womens | visit | T-learner | 0.27 ± 0.07 (per 100) | 6.91% | 4.59% |
| womens | visit | X-learner | 0.26 ± 0.06 (per 100) | 7.12% | 4.59% |
| womens | visit | Transformed outcome | 0.26 ± 0.04 (per 100) | 7.10% | 4.59% |
| womens | visit | random | -0.04 ± 0.05 (per 100) | 4.12% | 4.59% |
| womens | spend | S-learner | 28.54 ± 25.44 ($ per 1,000) | $0.766 | $0.503 |
| womens | spend | X-learner | 18.70 ± 16.85 ($ per 1,000) | $0.644 | $0.503 |
| womens | spend | Transformed outcome | 16.09 ± 35.53 ($ per 1,000) | $0.706 | $0.503 |
| womens | spend | T-learner | 5.78 ± 16.04 ($ per 1,000) | $0.588 | $0.503 |
| womens | spend | rule: bought this category | -2.30 ± 24.23 ($ per 1,000) | $0.457 | $0.503 |
| womens | spend | random | -14.85 ± 19.26 ($ per 1,000) | $0.353 | $0.503 |

## 6. Who should get which e-mail? (margin 30%; profit vs sending nothing, $ per 1,000 customers)

Mean over 5 held-out splits; the 95% CI is the paired bootstrap on split 0.

| Policy | $0.05 / e-mail | $0.15 / e-mail | $0.30 / e-mail | Share e-mailed ($0.15) |
|---|---|---|---|---|
| everyone: mens | +176 [+22, +305] | +76 [-78, +205] | -74 [-228, +55] | 100% |
| everyone: womens | +101 [+18, +334] | +1 [-82, +234] | -149 [-232, +84] | 100% |
| random arm | +91 [+19, +268] | +7 [-108, +134] | -60 [-195, +35] | 66% |
| rule: category bought | +120 [+37, +331] | +20 [-63, +231] | -130 [-213, +81] | 100% |
| model: spend | +132 [+42, +337] | +54 [-4, +228] | -31 [-90, +146] | 69% |
| model: visit proxy | +154 [-1, +267] | +48 [-104, +132] | -22 [-118, +54] | 78% |

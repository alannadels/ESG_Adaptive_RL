# Full-sweep results

Regime-indexed evolutionary reward discovery for a values-screened ESG allocator: 8 optimizers x 3 regime detectors (SMA crossover / EMA crossover / walk-forward HMM) x 3 regimes (bull/neutral/bear). Single-seed search, 10-seed evaluation.

## 1. Evolved reward pattern (mean weights across the 8 optimizers)

| detector | regime | w_return | w_e | w_s | w_g | w_risk | dominant |
|---|---|---|---|---|---|---|---|
| crossover | bull | 0.160 | 0.297 | 0.130 | 0.131 | 0.282 | E |
| crossover | neutral | 0.288 | 0.164 | 0.290 | 0.069 | 0.189 | S |
| crossover | bear | 0.309 | 0.118 | 0.198 | 0.147 | 0.229 | S |
| ema | bull | 0.120 | 0.407 | 0.043 | 0.095 | 0.336 | E |
| ema | neutral | 0.301 | 0.199 | 0.192 | 0.072 | 0.235 | E |
| ema | bear | 0.279 | 0.179 | 0.274 | 0.148 | 0.120 | S |
| hmm | bull | 0.140 | 0.343 | 0.152 | 0.169 | 0.197 | E |
| hmm | neutral | 0.207 | 0.157 | 0.183 | 0.222 | 0.231 | G |
| hmm | bear | 0.086 | 0.191 | 0.288 | 0.218 | 0.216 | S |

**Robustness — dominant pillar across all 24 cells (8 optimizers x 3 detectors):**
- **bull**: E=21/24, S=2/24, G=1/24
- **neutral**: S=14/24, E=8/24, G=2/24
- **bear**: S=14/24, E=6/24, G=4/24

## 2. Performance — evolved vs baselines, and vs SPY

| detector | regime | evolved Sharpe (±seed sd) | default | ret_only | ev-def | Sortino | beta_SPY | alpha_SPY | IR_SPY |
|---|---|---|---|---|---|---|---|---|---|
| crossover | bull | 1.27±0.02 | 1.26 | 1.25 | +0.01 | 1.91 | 0.77 | +0.048 | +0.12 |
| crossover | neutral | 0.51±0.08 | 0.50 | 0.50 | +0.01 | 0.74 | 0.83 | -0.047 | -0.96 |
| crossover | bear | 0.60±0.04 | 0.59 | 0.59 | +0.01 | 0.85 | 0.97 | -0.011 | -0.24 |
| ema | bull | 1.40±0.03 | 1.37 | 1.36 | +0.03 | 2.10 | 0.77 | +0.055 | +0.16 |
| ema | neutral | 0.46±0.04 | 0.45 | 0.44 | +0.01 | 0.62 | 0.88 | -0.084 | -1.61 |
| ema | bear | 0.31±0.03 | 0.31 | 0.30 | -0.00 | 0.46 | 0.94 | +0.041 | +0.65 |
| hmm | bull | 1.06±0.02 | 1.05 | 1.04 | +0.01 | 1.54 | 0.76 | +0.025 | -0.05 |
| hmm | neutral | 2.10±0.03 | 2.11 | 2.11 | -0.01 | 3.23 | 0.81 | +0.043 | -0.26 |
| hmm | bear | 0.05±0.02 | 0.05 | 0.04 | +0.00 | 0.08 | 0.91 | -0.002 | -0.04 |

*Evolved Sharpe is the mean over 10 PPO seeds then over 8 optimizers; the seed sd (~0.02-0.08) is comparable to the ev-def gap, so evolving the reward does not beat the naive baselines significantly. beta<1 throughout; positive SPY-alpha in bull and (mostly) bear, negative in neutral.*

## 3. ESG differential

Coverage: 50/50 universe names have ESG; 494 S&P 500 names benchmarked.

**Universe vs the full S&P 500 (Refinitiv, most recent fiscal year):**

| metric | our universe | S&P 500 | uplift | our %ile |
|---|---|---|---|---|
| Overall (0-100) | 78.3 | 64.7 | +13.6 | 86th |
| E (0-5) | 3.2 | 2.7 | +0.5 | 73th |
| S (0-5) | 3.0 | 2.4 | +0.7 | 76th |
| G (0-5) | 3.6 | 3.3 | +0.3 | 73th |

**Realized portfolio ESG exposure per regime (holdings-weighted, normalized 0-1):**

| regime | E | S | G | turnover |
|---|---|---|---|---|
| bull | 0.64 | 0.61 | 0.71 | 0.000 |
| neutral | 0.63 | 0.61 | 0.71 | 0.001 |
| bear | 0.63 | 0.60 | 0.70 | 0.001 |

*Exposure is high and near-constant across regimes at ~0 turnover — guaranteed by the screened universe, not the agent.*

**Alpha vs the market and dedicated ESG ETFs (annualized):**

| benchmark | bull | neutral | bear |
|---|---|---|---|
| SPY | +0.043 | -0.030 | +0.009 |
| SUSA | +0.046 | -0.015 | +0.007 |
| DSI | +0.055 | -0.032 | +0.015 |
| SPYX | +0.043 | -0.025 | +0.019 |

*Positive in bull and bear vs all four, including the ESG ETFs (SUSA/DSI/SPYX) — which we match/beat while screening harder (hard exclusions vs their best-in-class tilts).*

## 4. Timing

| optimizer | cells | mean_s | total_min |
|---|---|---|---|
| acor | 9 | 4584 | 688 |
| abc | 9 | 4440 | 666 |
| lshade | 9 | 2484 | 373 |
| de | 9 | 2482 | 372 |
| gwo | 9 | 2464 | 370 |
| pso | 9 | 2458 | 369 |
| cma | 9 | 2347 | 352 |
| xnes | 9 | 2317 | 348 |

Total core-time 3536 min (59 core-hours); wall-clock ~3h12m across 22 workers.

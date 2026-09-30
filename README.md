# ESG-Adaptive RL

**When Does Sustainability Pay? Regime-Indexed Evolutionary Reward Discovery for ESG Portfolio Allocation**

A reinforcement-learning framework for sustainable portfolio allocation in which the
trade-off between financial return, ESG factors, and tail risk is **evolved separately for
each market regime** (bull / neutral / bear), and a meta-controller switches between the
regime-specialist policies as market conditions change. The agent operates under a
**strict, no-compromise mandate**: harmful companies are excluded from its investable
universe entirely, so it can never hold them — in any regime.

> Research code for a paper targeting **ICAIF 2026** (ACM International Conference on AI in
> Finance). Status: **in development.**

---

## Motivation

ESG (Environmental, Social, Governance) investing faces a persistent, honest question:
*does directing capital toward sustainable firms cost you financially — and if so, when?*

Most existing ESG reinforcement-learning work collapses this into a single, **fixed**
weighting between return and ESG. But the optimal balance plausibly depends on the **market
regime**: sustainability may be affordable in calm, rising markets and require active
de-risking in downturns. This project makes that trade-off **explicit, regime-adaptive, and
interpretable** — and measures, rather than assumes, the financial cost of doing good.

## Core idea

Rather than hand-tuning how much an agent values return vs. E, S, G, and risk, we **evolve**
that weighting — and we evolve a **different weighting for each market regime**.

- **Reward weights** = the agent's *preferences* over `(return, E, S, G, tail-risk)`.
- **Portfolio weights** = the agent's *actions* (asset allocation), which it **learns** in
  order to score well under the current regime's preferences.
- An evolutionary outer loop searches the reward-weight space; a PPO inner loop trains the
  allocator for each candidate.
- **Two universes.** The agent's *investable* universe is ESG-eligible names only — harmful
  companies (fossil fuels, weapons, tobacco, ...) are removed from its action space
  entirely and can never be held, in any regime. Those names live only in a separate
  *benchmark* universe (the full market), used solely to measure what the strict mandate
  costs. Regime-adaptive E/S/G weighting operates *within* the values-compliant set.

The headline deliverable is a **finding**: the three evolved weight vectors, side by side,
reveal *which ESG factors matter in which regime* (e.g., "Governance and tail-risk dominate
in bear markets; Environmental and return dominate in bull markets").

## Research questions

1. **Primary (finding):** Which factors — return, E, S, G, tail risk — drive a successful
   sustainable portfolio, and how does their relative importance shift across bull, neutral,
   and bear regimes?
2. **Demonstration:** A **strict-exclusion** allocator that never holds a harmful company is
   made competitive by regime-adaptive E/S/G weighting. How close does it come to an
   *unconstrained* market benchmark, and what does the strict mandate cost (an honest,
   regime-conditional "price of virtue")?

## Contributions (honest scope)

- **Method (a new combination):** regime-indexed evolutionary discovery of a *multi-factor*
  `(return, E, S, G, risk)` reward weighting for an RL allocator.
- **Finding (primary novelty):** an empirical, regime-dependent ESG factor-importance
  ranking.
- **Demonstration:** a rigorous, transaction-cost-aware measurement of the regime-conditional
  price of virtue.

The regime-switching wrapper itself uses **standard, well-established machinery** (Markov
regime-switching) and is *not* claimed as a contribution — it is the vehicle that delivers
the finding.

---

## Architecture

```
                         ┌─────────────────────────────┐
                         │      Regime Detector         │
                         │  (HMM / transparent rules)   │
                         │   causal, no look-ahead      │
                         └──────────────┬──────────────┘
                                        │ regime ∈ {bull, neutral, bear}
                                        ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                       Meta-Strategy (switcher)                     │
   │   selects the active regime-specialist sub-policy                  │
   └───────┬───────────────────┬───────────────────────┬───────────────┘
           ▼                   ▼                        ▼
   ┌───────────────┐   ┌───────────────┐        ┌───────────────┐
   │  BULL policy  │   │ NEUTRAL policy│        │  BEAR policy  │
   │ evolved reward│   │ evolved reward│        │ evolved reward│
   │  weights w_b  │   │  weights w_n  │        │  weights w_r  │
   └───────────────┘   └───────────────┘        └───────────────┘
        each = PPO allocator trained under its regime's reward schedule
```

### Bi-level optimization (per regime)

```
Outer loop  (Evolutionary search: CMA-ES / DE / L-SHADE)
    proposes reward-weight vector  w = (w_return, w_E, w_S, w_G, w_risk)
        │
        ▼
Inner loop  (PPO)
    trains a portfolio allocator under reward:
        r_t = w_return·return_t + w_E·E + w_S·S + w_G·G − w_risk·tail_risk_t
        │
        ▼
Fitness = regime-specific risk-adjusted performance (out-of-sample)
        │
        └──► returns fitness to the evolutionary outer loop
```

The reward decomposes **ESG into its sub-factors (E / S / G)** so the evolved weights yield a
fine-grained, interpretable factor-importance result rather than a single opaque "ESG" knob.

---

## Underlying infrastructure

| Component | Tooling |
|---|---|
| RL algorithm | PPO (Stable-Baselines3) |
| Portfolio environment | Custom Gymnasium environment (minimal-dependency; full control of the reward seam) |
| Evolutionary search | CMA-ES, Differential Evolution, L-SHADE (`pycma`, `pymoo` / custom) |
| Regime detection | **Walk-forward 3-state Gaussian HMM (default)** — filtered posteriors, monthly expanding-window refits (`esg_regime`); transparent MA/trend rules remain available (`detector="crossover"/"trend"`) |
| Classical baseline | Riskfolio-Lib (static mean-CVaR / mean-variance efficient frontier) |
| Market data | `yfinance` / Stooq (free daily prices) |
| ESG data | Public ESG sub-scores where available; proxies otherwise (carbon intensity → E, controversy counts → S, board-independence metrics → G) |
| Language / stack | Python, NumPy, pandas, PyTorch |

### Repository structure

Currently implemented (the v0 backbone):

```
ESG_Adaptive_RL/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── train_single.py            # entry point: train + evaluate one fixed-weight allocator
├── evolve_regimes.py          # entry point: per-regime 8-optimizer reward-weight evolution
├── evolve_regimes_compare.py  # entry point: per-regime evolution under HMM vs crossover labels
├── evolve_meta.py             # entry point: meta-controller backtest over per-regime specialists
├── build_membership_history.py # point-in-time S&P 500 membership spans (Wikipedia revision samples)
├── build_universe_snapshots.py # year-stamped, look-ahead-free universe re-selection per year
└── esg_adaptive_rl/
    ├── __init__.py
    ├── config.py              # 50-name screened universe, dates, lookback, reward weights, PPO settings
    ├── data.py                # price download + ESG tables (synthetic or real) + regime-CSV loader
    ├── esg_data.py            # real Refinitiv ESG loader (annual→daily, publication lag)
    ├── reward.py              # configurable multi-factor reward (the swappable seam)
    ├── env.py                 # custom Gymnasium portfolio environment (look-ahead-safe)
    ├── meta.py                # MetaController: regime-switching over per-regime specialist policies
    ├── metrics.py             # return, Sharpe, CVaR, drawdown, ESG profile
    └── regimes.py             # causal regime labeling: HMM (default) / 50-200 crossover / trend
```

Planned modules (not yet implemented):

```
baselines/   # Riskfolio frontier, 60/40, plain HMM-regime allocator
eval/        # significance tests, crisis-window robustness, figures
```

---

## Results so far — regime detection (validated, out-of-sample)

The regime layer is built and validated. All results are **point-in-time**: features
are standardized on lagged expanding windows, the HMM is refit monthly on an expanding
window and decoded with *filtered* (forward-only) posteriors, and every overlay signal
is lagged one day. Full write-ups live in
[`esg_regime/results/`](esg_regime/results/) (`RESULTS.md`,
`INDEX_COMPARISON.md`, `HEURISTICS_BENCHMARKS.md`, `METHODOLOGY.md`).

**The regime labels are economically real.** Grouping next-day returns by the label
assigned the prior day, stress-state days carry ~2-3x the volatility of calm days on
every universe tested — 3 ESG ETFs (SUSA, DSI, ESGU), 3 broad benchmarks
(S&P 500, SPY, QQQ), and 11 published ESG indices in an earlier 12-universe study:

| Universe | Calm vol | Stress vol | Ratio |
|---|---:|---:|---:|
| SPY | 11.6% | 35.0% | **3.02** |
| ESGU | 12.1% | 34.1% | 2.82 |
| SUSA | 11.5% | 31.2% | 2.70 |

**A regime overlay ~halves risk out-of-sample.** A naive 100/60/0 validation overlay
(calm/choppy/stress, net of costs) cut max drawdown on 11 of 11 published ESG indices
(average 54%; e.g. DSI test-period Sharpe 0.92 vs 0.75 buy-and-hold, max drawdown
-7.7% vs -28.4%).

**The HMM default is evidence-based.** An 11-detector comparison (HMM + 10 causal
rule-based detectors from a 27-method literature catalog) across 6 universes ranks the
HMM and a simple expanding vol-percentile rule at the top (stress/calm separation 2.73
/ 2.85, overlay dSharpe +0.11 / +0.19); the 50/200 crossover, while the most *stable*
label, is the weakest risk separator (1.70). Consecutive-down-day streak rules turn
out to be short-term *reversal* signals, not regimes (next-day returns after 5 straight
down days: +68% to +164% annualized).

**Detecting regimes on SPY transfers to broad ESG universes.** SPY-detected labels
agree 85-96% with each broad ESG universe's own labels and separate volatility equally
well — validating `REGIME_INDEX = "SPY"`. (Caution remains for narrow/thematic
universes, e.g. clean-energy.)

---

## Implementation status

This is an actively developing research repository. The lists below separate what the
current code does from what remains to be built.

### Implemented (v0 backbone)

- Custom Gymnasium portfolio-allocation environment: long-only, fully invested, with
  transaction costs and look-ahead-safe timing (the action never sees the same-day
  return it is graded on).
- Configurable multi-factor reward over `(return, E, S, G, tail-risk)`, with the weight
  vector exposed as an explicit, swappable object — the seam every later stage plugs into.
- Time-indexed placeholder ESG table (deterministic synthetic data), built so a real
  historical ESG history can replace it with no change to the environment or agent.
- Free daily price loader (yfinance) with a chronological train/test split.
- Evaluation metrics: annualized return, Sharpe, CVaR, maximum drawdown, realized ESG
  profile, and turnover.
- Single fixed-weight PPO training-and-evaluation entry point (`train_single.py`).
- The backbone is verified end-to-end on synthetic data: Gymnasium API compliance, the
  reward, the metrics, and a PPO training loop all run.
- **Causal market-regime detection** (`esg_regime` + `esg_adaptive_rl/regimes.py`):
  a walk-forward 3-state Gaussian HMM (the default detector) plus ten rule-based
  alternatives (50/200 crossover, trend, 20% drawdown rule, vol-percentile, VIX
  cutoffs, TSMOM, Lunde-Timmermann, ...), all validated look-ahead-free — see the
  Results section below.
- Rule-based regime labeling, per-regime evolutionary search over reward weights
  (eight nature-inspired optimizers), and a real ESG loader (`evolve_regimes.py`).
- HMM-vs-crossover per-regime RL training comparison (`evolve_regimes_compare.py`),
  feeding the specialists' evolved reward weights into a live-switching
  **regime-switching meta-controller** (`esg_adaptive_rl/meta.py` + `evolve_meta.py`):
  the switcher applies the label-appropriate specialist policy every day (causal
  label of day `t-1`, whipsaw/turnover accounted) and backtests the whole system
  against single-policy and 1/N baselines on the 2021+ window.
- **Point-in-time universe snapshots** (`build_membership_history.py` +
  `build_universe_snapshots.py`): yearly top-5-per-sector re-selection from the
  S&P 500's year-end constituent tables (Wikipedia revision samples) with
  publication-lag-correct ESG/return windows, replacing the retrospective fixed
  universe (2017–2025 scores traded over 2005–2026) for universe construction —
  no hindsight survivors, no look-ahead; first snapshot year 2008.

### Not yet implemented (planned)

- Real, look-ahead-free historical ESG / impact data to replace the placeholder table.
- Per-regime evolved schedules finalized for the three regimes (bull / neutral / bear).
- Baselines: the static convex ESG–CVaR frontier
  (Riskfolio-Lib), equal-weight, 60/40, and a plain HMM-regime allocator.
- Full evaluation: multiple seeds, significance tests, a crisis-window stress test, and
  the headline factor-importance figure.

---

## Baselines

- Return-only RL allocator
- Fixed-weight (non-regime) ESG RL allocator
- Static convex return–ESG–CVaR efficient frontier (Riskfolio-Lib)
- Plain HMM-regime allocator (no evolved ESG schedule)
- Equal-weight and 60/40 / buy-and-hold

## Evaluation metrics

Reported **per regime and overall**, with significance tests across seeds:

- Return, Sharpe ratio
- **CVaR / tail risk**, max drawdown, return-to-tail ratio
- Portfolio ESG profile (E / S / G)
- Turnover and **transaction costs**, regime-switch frequency (whipsaw)

## Methodological safeguards (non-negotiable)

- **No look-ahead bias:** regime labels at time *t* use only information available up to *t*.
- **Transaction costs charged** in all backtests; switching frequency reported.
- **Multiple seeds + significance tests**; evaluation spans ≥ 2 market periods including a
  crisis window.
- Honest reporting of the price of virtue — including null or negative results.

---

## Roadmap

- [x] v0 backbone: custom env, multi-factor reward seam, placeholder ESG, metrics, and a
  single fixed-weight PPO pipeline (validated end-to-end on synthetic data)
- [ ] Real, look-ahead-free historical ESG / impact data -- Ash
- [ ] Evolutionary outer loop (CMA-ES / DE / L-SHADE) over the reward-weight vector -- Alan
- [ ] Causal regime labeling + detector -- Satya
- [ ] Per-regime evolved schedules (bull / neutral / bear) and the factor-importance figure
- [x] Regime-switching meta-controller (specialists + causal switcher, `esg_adaptive_rl/meta.py`
  + `evolve_meta.py`; first backtest pending the HPC run) -- Gurjot
- [ ] Baselines + full evaluation (multiple seeds, significance, crisis-window robustness) -- Gurjot
- [ ] Paper write-up + reproducibility release

## Author

Alan Nadelsticher Ruvalcaba — Georgia Institute of Technology · University of Pennsylvania

## License

Released under the [MIT License](LICENSE).

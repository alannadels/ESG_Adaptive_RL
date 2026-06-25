# ESG-Adaptive RL

**When Does Sustainability Pay? Regime-Indexed Evolutionary Reward Discovery for ESG Portfolio Allocation**

A reinforcement-learning framework for sustainable portfolio allocation in which the
trade-off between financial return, ESG factors, and tail risk is **evolved separately for
each market regime** (bull / neutral / bear), and a meta-controller switches between the
regime-specialist policies as market conditions change.

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

The headline deliverable is a **finding**: the three evolved weight vectors, side by side,
reveal *which ESG factors matter in which regime* (e.g., "Governance and tail-risk dominate
in bear markets; Environmental and return dominate in bull markets").

## Research questions

1. **Primary (finding):** Which factors — return, E, S, G, tail risk — drive a successful
   sustainable portfolio, and how does their relative importance shift across bull, neutral,
   and bear regimes?
2. **Demonstration:** Under a regime-adaptive framework, is an ESG portfolio financially
   competitive with (and more resilient under stress than) return-only and fixed-weight
   baselines — and what does sustainability actually cost *when* it costs (an honest,
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
| Regime detection | Hidden Markov Model (Hamilton-style regime switching) or transparent trend/drawdown rules |
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
└── esg_adaptive_rl/
    ├── __init__.py
    ├── config.py              # universe, dates, lookback, reward weights, PPO settings
    ├── data.py                # price download + time-indexed placeholder ESG table
    ├── reward.py              # configurable multi-factor reward (the swappable seam)
    ├── env.py                 # custom Gymnasium portfolio environment (look-ahead-safe)
    └── metrics.py             # return, Sharpe, CVaR, drawdown, ESG profile
```

Planned modules (not yet implemented):

```
evolution/   # CMA-ES / DE / L-SHADE outer loop over the reward-weight vector
regimes/     # causal market-regime labeling + HMM / rule-based detector
meta/        # regime-switching meta-controller over the per-regime policies
baselines/   # return-only RL, fixed-weight RL, Riskfolio frontier, 60/40, equal-weight
eval/        # significance tests, crisis-window robustness, figures
```

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

### Not yet implemented (planned)

- Real, look-ahead-free historical ESG / impact data to replace the placeholder table.
- Evolutionary outer loop (CMA-ES / DE / L-SHADE) that searches the reward-weight vector.
- Causal market-regime labeling and a detector (HMM or transparent trend/drawdown rules).
- Per-regime evolved schedules for the three regimes (bull / neutral / bear).
- The regime-switching meta-controller over the per-regime policies.
- Baselines: return-only RL, fixed-weight RL, the static convex ESG–CVaR frontier
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
- [ ] Real, look-ahead-free historical ESG / impact data
- [ ] Evolutionary outer loop (CMA-ES / DE / L-SHADE) over the reward-weight vector
- [ ] Causal regime labeling + detector
- [ ] Per-regime evolved schedules (bull / neutral / bear) and the factor-importance figure
- [ ] Regime-switching meta-controller
- [ ] Baselines + full evaluation (multiple seeds, significance, crisis-window robustness)
- [ ] Paper write-up + reproducibility release

## Author

Alan Nadelsticher Ruvalcaba — Georgia Institute of Technology · University of Pennsylvania

## License

Released under the [MIT License](LICENSE).

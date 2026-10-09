# Ablations

Three ablations, all pointing the same way: **performance is driven by the values-screened universe, not the RL / reward / regime machinery.** The evolved, regime-indexed reward earns its place on *interpretability* (which ESG pillar is priced in which regime), not on returns.

## 1. Reward-component breakdown (crossover, Sharpe)

| reward | bull | neutral | bear |
|---|---|---|---|
| return alone | 1.25 | 0.50 | 0.59 |
| risk alone | 1.25 | 0.51 | 0.59 |
| ESG alone | 1.26 | 0.50 | 0.59 |
| return + risk | 1.25 | 0.50 | 0.59 |
| all (default) | 1.26 | 0.50 | 0.59 |
| evolved | 1.27 | 0.51 | 0.60 |

*Every reward formulation — including ESG-alone and risk-alone — gives essentially identical Sharpe (within ~0.02). No component or combination moves performance: the agent converges to ~equal-weight regardless of the reward, because the screened universe is uniformly high-ESG and low-dispersion.*

## 2. Naive allocation baselines (crossover, Sharpe)

| regime | equal-weight (1/N) | buy-and-hold | RL evolved |
|---|---|---|---|
| bull | 1.25 | 1.13 | 1.27 |
| neutral | 0.50 | 0.44 | 0.51 |
| bear | 0.59 | 0.56 | 0.60 |

*The RL allocator beats equal-weight by only ~0.01-0.02 Sharpe — within the seed noise. Buy-and-hold is slightly worse (drift). The learned, regime-indexed, evolved-reward allocator is, for practical purposes, 1/N of the screened universe.*

## 3. Regime-indexed vs global reward schedule (Sharpe, mean over 8 optimizers)

| detector | regime | regime-indexed | global | idx - global |
|---|---|---|---|---|
| crossover | bull | 1.270 | 1.266 | +0.004 |
| crossover | neutral | 0.508 | 0.495 | +0.013 |
| crossover | bear | 0.598 | 0.596 | +0.002 |
| ema | bull | 1.396 | 1.387 | +0.008 |
| ema | neutral | 0.456 | 0.454 | +0.002 |
| ema | bear | 0.307 | 0.300 | +0.008 |
| hmm | bull | 1.060 | 1.057 | +0.003 |
| hmm | neutral | 2.104 | 2.097 | +0.007 |
| hmm | bear | 0.051 | 0.048 | +0.003 |

*Regime-indexing beats a single global schedule by only +0.002 to +0.013 Sharpe — within seed noise, though consistently positive in all 9 cells. Its value is the discovered pattern (which pillar is priced when), not returns.*

## Summary

Across all three ablations the performance spread is within seed noise: no reward component, no learned allocator, and no regime-indexing meaningfully beats naive 1/N of the screened universe. This is the honest, reviewer-proof framing — the contribution is the *finding* and the *values-aligned, lower-beta, market-comparable* universe, not a performance win.

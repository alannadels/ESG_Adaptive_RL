# RL training under HMM vs crossover regime labels

End-to-end run of the project's core experiment with the regime layer plugged in:
regime labels -> per-regime dataset split -> CMA-ES over the reward weighting ->
PPO allocator -> validation Sharpe. Run twice, once per regime labeling, so the
labeling's effect on what the RL learns is measurable.

Reproduce: `python build_regime_datasets_hmm.py && python evolve_regimes_compare.py`
Raw: `rl_regime_comparison.json`, `rl_regime_comparison.txt`.

## Setup

| | |
|---|---|
| Universe | the 50-name screened ESG universe (`config.UNIVERSE`), real Refinitiv ESG |
| Regime index | SPY, 2005-01-04 -> 2026-08-07 (5,432 trading days) |
| Labelings | **hmm** = walk-forward 3-state Gaussian HMM · **crossover** = causal 50/200 SMA |
| Split | chronological 70/30 train/validation *within* each regime subset |
| Search | CMA-ES, population 10 x 10 generations, 10k PPO steps per candidate |
| Final eval | evolved weights re-scored at 30k PPO steps vs two baselines |
| Objective | validation Sharpe (purely financial, so evolved E/S/G weights reflect what helps risk-adjusted return) |

**Both labelings emit `bull`/`neutral`/`bear`**, so the same `PortfolioEnv` and
search consume either with no translation (the label vocabularies were unified
for exactly this reason).

## Regime day-counts — the HMM balances the buckets better

| Labeling | bull | neutral | bear |
|---|---:|---:|---:|
| hmm | 2,794 (51%) | **1,808 (33%)** | 830 (15%) |
| crossover | 3,762 (69%) | 846 (16%) | 824 (15%) |

The crossover leaves the neutral specialist only 846 days (~592 after the 70/30
split). The HMM more than doubles that to 1,808, which matters directly for
per-regime training.

## Evolved reward weights (normalized)

| Labeling | Regime | fitness | w_return | w_E | w_S | w_G | w_risk |
|---|---|---:|---:|---:|---:|---:|---:|
| crossover | bull | 1.281 | 0.396 | **0.499** | 0.010 | 0.061 | 0.033 |
| crossover | neutral | 0.524 | 0.056 | 0.286 | 0.226 | 0.073 | **0.359** |
| crossover | bear | 0.608 | 0.169 | 0.125 | **0.328** | 0.277 | 0.101 |
| hmm | bull | 1.054 | 0.224 | 0.349 | 0.032 | 0.027 | **0.368** |
| hmm | neutral | 2.115 | **0.441** | 0.092 | 0.135 | 0.001 | 0.332 |
| hmm | bear | 0.074 | 0.262 | 0.018 | **0.401** | 0.002 | 0.317 |

## Validation Sharpe — evolved vs baselines (same window, 30k PPO steps)

| Labeling | Regime | evolved | default | return-only | evolved − default |
|---|---|---:|---:|---:|---:|
| crossover | bull | 1.290 | 1.256 | 1.241 | +0.034 |
| crossover | neutral | 0.482 | 0.521 | 0.493 | **−0.039** |
| crossover | bear | 0.616 | 0.587 | 0.594 | +0.028 |
| hmm | bull | 1.093 | 1.046 | 1.046 | +0.047 |
| hmm | neutral | 2.120 | 2.109 | 2.106 | +0.011 |
| hmm | bear | 0.055 | 0.036 | 0.043 | +0.018 |

## Findings

### 1. The regime layer works end to end
HMM labels flow into the RL with no translation, split cleanly into three
trainable subsets, and the evolutionary search runs per regime. This is the
project's intended architecture, executing.

### 2. HMM labeling gives consistent (small) gains; the crossover does not
Evolved weights beat `DEFAULT_WEIGHTS` in **3/3 HMM regimes** but only **2/3
crossover regimes** — the crossover's neutral cell is *worse* than the default
(−0.039), plausibly because its 846-day bucket leaves too little to learn from.

### 3. The bear regimes are genuinely different objects
HMM bear validation Sharpe is 0.055 vs the crossover's 0.616. The HMM's bear is
a high-volatility state (32% annualized); the crossover's is a *trend* state
that includes lagged recoveries. The HMM bucket is the harder, more honest test
of de-risking behaviour — and the one where regime-conditional risk weighting
should matter most.

### 4. The headline factor-importance claim is not yet robust — this is the
important caveat
The evolved weight vectors **disagree substantially between the two labelings**
for the same regime name. In bull, the crossover says E dominates (0.499, risk
0.033) while the HMM says risk dominates (0.368, E 0.349). Only the bear regime
agrees across labelings (S highest under both: 0.328 / 0.401).

So "which ESG factor matters in which regime" is, at this search budget,
**partly an artifact of how the regime was defined**. Before that becomes a
paper headline it needs:
- multiple seeds (everything here is a single seed) and a larger
  population/generation budget — the current search is deliberately reduced;
- significance testing on the Sharpe deltas, which are +0.011 to +0.047 and
  almost certainly inside noise at one seed;
- the full eight-optimizer sweep (`evolve_regimes.py`) so the result is not an
  artifact of CMA-ES alone.

The one finding stable across both labelings — **Social weight highest in the
bear regime** — is the candidate worth pursuing first.

### 5. Evolved weights barely beat return-only
`return_only` scores within ~0.05 Sharpe of both evolved and default everywhere.
On this universe and objective, the E/S/G weights are close to a free parameter:
the strict-exclusion universe may already capture most of the ESG effect,
leaving little for the weighting to add. That is itself a publishable, honest
result about the price of virtue under a values-compliant mandate.

---

# Part 2 — Head-to-head on identical days (the meta-controller)

Part 1 scores each labeling *inside its own regime buckets*, which hold different
dates — so `hmm neutral` (Sharpe 2.12) and `crossover neutral` (0.48) are not
comparable. This part runs the architecture the project proposes, which is:

1. split the timeline at **2019-01-01** (test spans COVID-2020 and the 2022 bear);
2. per labeling, train one PPO specialist per regime on that labeling's TRAIN days;
3. walk the **same 1,910 test days**, letting each labeling's point-in-time label
   choose which specialist acts;
4. score the single combined equity curve.

Reward weights are held fixed across all specialists, so the only difference
between the two arms is the labeling. Three seeds, averaged.
Reproduce: `python meta_controller_eval.py`.

| Labeling | TRAIN bull/neutral/bear | TEST bull/neutral/bear |
|---|---|---|
| hmm | 1,838 / 1,114 / 570 | 956 / 694 / 260 |
| crossover | 2,348 / 654 / 520 | 1,414 / **192** / 304 |

## Result: no measurable difference — between labelings, or from no regimes at all

| Variant | CAGR | Vol | Sharpe | MaxDD | CVaR5 |
|---|---:|---:|---:|---:|---:|
| meta_hmm | 15.47% | 19.68% | 0.830 | −37.00% | −2.86% |
| meta_crossover | 15.28% | 19.52% | 0.827 | −36.26% | −2.84% |
| **no_regime** (single allocator) | 15.89% | 19.55% | **0.853** | −36.47% | −2.84% |
| equal_weight | 15.25% | 19.60% | 0.823 | −36.69% | −2.86% |

Sharpe by seed — hmm `[0.835, 0.836, 0.818]`, crossover `[0.826, 0.816, 0.838]`.
**Mean difference: +0.003.** Seed-to-seed spread within one method (~0.018) is six
times the difference between methods, and the crossover wins on seed 2. There is
no signal here.

Worse for the architecture: the **no-regime single allocator scores highest**, and
every variant lands within 0.03 Sharpe of plain equal-weight.

## Why: the allocator is equal-weight in disguise

Diagnostic on a trained allocator rolled over the test window (50 assets):

| Measure | Value | Equal-weight reference |
|---|---:|---:|
| Mean max weight | 0.0282 | 0.0200 |
| Mean min weight | 0.0128 | 0.0200 |
| Effective N positions | **48.1** | 50.0 |
| Mean daily turnover | 0.0009 | 0 |
| **Corr. with equal-weight returns** | **0.9993** | 1.0 |

The policy is a near-uniform softmax that barely trades. Regime labels select
between three allocators that are all effectively the same portfolio, so **no
labeling can show an effect** — there is nothing for the regime signal to act
through. This also explains Part 1's tiny margins (+0.011 to +0.047) and why
evolved weights barely beat return-only.

## What this means

**The bottleneck is the allocator, not the regime labels.** The regime layer is
independently validated (2–3x stress/calm volatility separation out-of-sample,
54% drawdown reduction in the overlay backtest), but the current PPO setup cannot
express a differentiated portfolio, so none of that reaches the P&L.

Likely causes, in order of suspicion:
1. **Action parameterization.** A softmax over 50 raw scores starts uniform, and
   the reward difference between two near-uniform long-only portfolios of 50
   correlated large-caps is minute — so the gradient pushing away from uniform is
   very weak.
2. **Reward scale.** Daily net returns are ~5e-4; the ESG terms are O(1) but
   nearly constant across portfolios, so little differentiates candidates.
3. **Turnover cost** (10 bps per unit turnover) actively penalizes deviating.
4. **Training budget** (120k steps/specialist) — real, but secondary to the above.

**Recommended next steps before any further regime work:**
- Sanity-check that PPO beats a *random* policy at all on this env; if not, the
  issue is the env/reward, not the hyperparameters.
- Rescale or standardize the reward, and consider allowing concentration (e.g.
  temperature on the softmax, or top-k weights).
- Re-run this exact head-to-head afterwards — it is cheap and is the honest test
  of whether regime conditioning buys anything.

Until an allocator can differentiate, comparing regime detectors on downstream RL
performance cannot discriminate between them. The right comparison of detectors
today is the direct one in `HEURISTICS_BENCHMARKS.md`, where the HMM does separate
volatility states 2.73x vs the crossover's 1.70x.

---

## Budget caveat

CMA-ES only, population 10 x 10 generations, 10k PPO steps per candidate, one
seed. The full protocol in `UNIVERSE_AND_TRAINING.md` (eight optimizers,
population 12 x 15 generations, 20k steps) is ~86M PPO timesteps across the
grid — days of CPU. Treat every number here as directional, not final.

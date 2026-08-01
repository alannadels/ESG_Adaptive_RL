# Heuristic regimes vs the HMM, and ESG vs broad benchmarks

Grades 11 regime detectors — the walk-forward HMM plus 10 causal rule-based
heuristics, including the repo's production 50/200 crossover and a
user-suggested consecutive-down-days rule — on 3 ESG universes (SUSA, DSI,
ESGU) and 3 broad benchmarks (^GSPC, QQQ, SPY), 2005–2026. All detectors are
point-in-time; all are graded by the same harness (next-day vol separation,
100/60/0 overlay net of costs, label churn). Also runs the **transfer test**:
regimes detected on SPY applied to ESG universes — exactly what
`evolve_regimes.py` (`REGIME_INDEX = "SPY"`) does.

Reproduce: `python -m esg_regime.benchmark_compare`
Raw grids: `heuristics_benchmarks.csv`, `transfer_test.csv`, `heuristics_output.txt`.

## The detectors

| Detector | Rule | Origin |
|---|---|---|
| `hmm` | 3-state Gaussian HMM, walk-forward | this package |
| `ma_crossover` | 50/200 SMA, 2% band, 10d dwell | **repo production** (`esg_adaptive_rl.regimes`) |
| `trend_200` | price vs 200 SMA, 2% band | repo baseline |
| `drawdown_bear` | ≥20% off peak = bear; ≥10% = correction | industry convention |
| `consec_down` | 5 straight down closes → stress; 3–4 → choppy | user-suggested family |
| `downday_frac` | ≥7 of last 10 days down → stress | smoother streak variant |
| `vol_percentile` | 20d realized vol vs own expanding 60th/85th pctile (through t−1) | vol-regime literature |
| `vix_threshold` | VIX <20 / 20–30 / >30 | classic practitioner cutoffs |
| `mom_12m` | 252d return > +5% / < −5% | time-series momentum (Moskowitz-Ooi-Pedersen 2012) |
| `lunde_timmermann` | first-passage filter, 20%/15% thresholds | Lunde & Timmermann (2004), the *causal* academic bull/bear rule |
| `ret_sign_60` | trailing 60d return sign, ±5% band | the deep-RL trading literature's bull/neutral/bear convention |

Retrospective dating algorithms (Pagan-Sossounov 2003, Bry-Boschan/BBQ, NBER
dating) are excluded by design: they date peaks/troughs using data *after* the
turning point, so they cannot give honest point-in-time labels.

## Average scores across all six universes

| Detector | Vol ratio | ΔSharpe | DD cut | Stress-bucket ann. fwd return | Switches/yr |
|---|---:|---:|---:|---:|---:|
| vix_threshold | **3.74** | −0.02 | 47% | +41% | 19.5 |
| vol_percentile | 2.85 | **+0.19** | **67%** | +9% | 11.1 |
| hmm | 2.73 | +0.11 | 63% | +12% | 9.2 |
| ret_sign_60 | 2.70 | −0.11 | 40% | +35% | 24.5 |
| drawdown_bear | 2.64 | −0.08 | 37% | +52% | 7.2 |
| downday_frac | 2.24 | −0.16 | 18% | +48% | 46.8 |
| mom_12m | 2.21 | −0.07 | 37% | +29% | 9.2 |
| lunde_timmermann | 2.10 | −0.02 | 44% | +22% | **0.8** |
| consec_down | 1.94 | −0.14 | −6% | **+122%** | 26.3 |
| trend_200 | 1.92 | −0.05 | 33% | +22% | 1.7 |
| ma_crossover | 1.70 | +0.03 | 34% | +10% | 2.2 |

## Findings

### 1. A one-line vol-percentile rule ≈ the HMM
`vol_percentile` matches or beats the HMM on separation (avg 2.85 vs 2.73) and
overlay economics (+0.19 vs +0.11 ΔSharpe), agrees with it 71–82% of days, and
needs no fitting at all. The HMM's remaining edge is label stability (9.2 vs
11.1 switches/yr) and joint modeling of trend/drawdown alongside vol. Honest
conclusion: **most of what the HMM finds is recoverable from realized vol
alone** — the sophisticated model buys polish, not a different signal.

### 2. The repo's production detectors are the weakest *risk* separators
`ma_crossover` has the lowest stress/calm vol ratio on nearly every universe
(avg 1.70; on ESGU just 1.21 with a 0% drawdown cut), `trend_200` the
second-lowest (1.92). This is not a bug — they are **direction** rules: their
bear states capture regimes of falling prices, not high risk, and they are by
far the most stable labels (1.7–2.2 switches/yr, valuable for training
specialist allocators). But if the point of regime-conditional reward weights
is to adapt the *risk* trade-off (`w_risk`) by market state, a vol-based state
definition is materially stronger.

### 3. The consecutive-down-days rule is a reversal signal, not a regime
The user-suggested rule (5 straight down days → bear) fails on three counts:
- **Too rare** — stress is 1.2–1.4% of days (31–77 days per universe over ~20
  years). Nowhere near enough days to train a per-regime specialist.
- **Too whippy** — ~27 label switches/yr.
- **Backwards** — next-day returns after 5 straight down days are strongly
  POSITIVE: +68% to +164% annualized. That is the classic short-term reversal
  effect: 5-down-day streaks end in bounces. De-risking there sells the bottom;
  the overlay loses on every universe (ΔSharpe −0.11 to −0.16).

As an *alpha* signal (buy the streak) it is interesting — but that is the
opposite of its proposed use as a bear flag.

### 4. Broad benchmarks regime just as well — ESG regimes are market regimes
The HMM's vol ratio on SPY (3.02) and ^GSPC (2.96) *exceeds* every ESG ETF
(2.31–2.82); QQQ is 2.57. Nothing in broad-ESG price dynamics is
regime-special: these universes are high-correlation subsets of the US large-cap
market, and their risk states are the market's risk states.

### 5. Transfer test: the `REGIME_INDEX = "SPY"` design is validated (for broad ESG)
SPY-detected labels applied to ESG universes (the repo's exact as-of alignment)
agree with the universes' own labels **85–96% of days** and give equal or
*better* vol separation (ESGU HMM: 3.33 via SPY labels vs 2.82 via its own):

| Target | Detector | Own vol ratio | SPY vol ratio | Own ΔSh | SPY ΔSh | Agreement |
|---|---|---:|---:|---:|---:|---:|
| SUSA | hmm | 2.70 | 2.80 | +0.13 | +0.08 | 91% |
| SUSA | vol_percentile | 2.82 | 2.83 | +0.18 | +0.21 | 92% |
| ESGU | hmm | 2.82 | **3.33** | +0.23 | +0.07 | 86% |
| ESGU | drawdown_bear | 3.02 | **4.07** | −0.06 | +0.01 | 96% |
| ESGU | vol_percentile | 2.98 | 3.11 | +0.30 | +0.20 | 85% |

This **refines the earlier caution** from the 12-index study ("detect on the
universe you trade", based on 43–67% cross-index agreement): that low agreement
was driven by the defective constructed basket and by *thematic* universes
(clean-energy ICLN). For **broad ESG universes, SPY-based regime detection is
sound** — the repo's design choice holds. The caution stands only for
narrow/thematic ESG portfolios.

### 6. VIX separates best but trades worst; and no stress bucket has negative mean returns
`vix_threshold` is the sharpest raw classifier everywhere (3.7–4.1) — implied
vol is genuinely forward-looking — yet its overlay ΔSharpe ≈ 0: VIX>30 days
include the violent rebounds (+41% ann. mean), so binary de-risking on VIX
gives back what it saves. Its right role is as a *feature* (as in the original
SPY reference model), not a standalone switch. More broadly, **no detector's
stress bucket has negative mean forward returns** — regime de-risking pays via
variance reduction, not return timing.

### 7. Two literature additions bracket the design space
- **Lunde-Timmermann (2004)** — the *causal* academic bull/bear filter — is the
  most stable label in the grid by an order of magnitude (0.8 switches/yr) with
  a respectable 2.10 vol ratio and ~16% bear days: the best choice when label
  *stability* and a large, coherent bear training set matter more than sharp
  risk timing.
- **ret_sign_60** — the convention deep-RL trading papers themselves use —
  separates vol well (2.70) but flips ~25×/yr and labels recoveries as bears
  (stress-bucket mean +35% ann.): a caution for RL work that borrows it
  uncritically.

## The researched method catalog

The full literature sweep (27 methods with rules, parameters, causality flags
and citations — from NBER dating and Dow Theory through Hamilton (1989),
Pagan-Sossounov, Lunde-Timmermann, GARCH/MS-GARCH, VIX conventions,
Kritzman-Li turbulence, TSMOM, Baz MACD, Moreira-Muir, to the DRL-paper
conventions) is saved at `regime_methods_catalog.json`. Two things it
confirms about this study's design:

- **Filtered-not-smoothed + expanding refits is the one honest HMM protocol.**
  The catalog flags full-sample smoothed probabilities as "the single most
  common look-ahead bug" in this literature — precisely what the walk-forward
  filtered decode here avoids.
- **Consecutive-down-day streaks are documented as an entry trigger, not a
  regime** (Connors mean-reversion setups) — independently matching finding 3.

## Recommendations for `evolve_regimes.py`

1. **Switch (or augment) the detector.** Replace the 50/200 crossover with the
   vol-percentile rule (one line, causal, no fitting) or the HMM for the
   regime axis that conditions `w_risk`. If directional specialisation is also
   wanted, use *two* axes: trend (crossover) × vol state.
2. **Keep `REGIME_INDEX = "SPY"`** — validated for broad-ESG universes (85–96%
   transfer agreement). Revisit only if the universe becomes thematic.
3. **Mind the bear-sample size.** Even good detectors put only 10–20% of days
   in stress; rules like consec-down (1.3%) leave a specialist allocator with
   ~70 training days — check day-counts before trusting per-regime evolution.

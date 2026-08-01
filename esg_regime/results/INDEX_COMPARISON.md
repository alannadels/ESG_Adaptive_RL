# Published ESG indices vs the score-constructed basket

**Question.** Does the regime HMM need a bespoke universe built from ESG scores, or
does fitting the same model directly on an off-the-shelf ESG index work as well?

**Method.** Identical pipeline on all 12 series (same 6 features, same 3-state HMM,
same walk-forward monthly retrain, same overlay and costs). Scored on regime quality
(OOS stress/calm volatility ratio), economic value (overlay vs buy & hold), and
label agreement with the constructed basket.

## Headline

| | Vol ratio (stress/calm) | ΔSharpe | Drawdown cut |
|---|---:|---:|---:|
| Score-constructed basket | 2.00 | **−0.51** | 53% |
| Published ESG indices (11) | **2.41** (median 2.31, range 1.68–2.92) | **+0.04** | 54% |

**The published indices beat the constructed basket** on both regime quality and
economics. The bespoke, ESG-score-built universe was *not* necessary — and is in
fact the weakest performer on ΔSharpe in the whole panel.

## Regime quality — vol separation by index

| Index | Style | Bars | Calm vol | Stress vol | Ratio | Stress % |
|---|---|---:|---:|---:|---:|---:|
| ESGV | broad ESG | 1657 | 13.0% | 38.0% | **2.92** | 17% |
| NULV | ESG value | 2084 | 11.2% | 32.5% | **2.91** | 15% |
| ESGU | broad ESG | 2107 | 12.1% | 34.1% | **2.82** | 19% |
| SPYX | fossil-fuel free | 2363 | 11.2% | 30.5% | 2.71 | 21% |
| SUSA | broad ESG | 5085 | 11.5% | 31.2% | 2.70 | 19% |
| CRBN | low carbon (global) | 2607 | 11.0% | 25.4% | 2.31 | 24% |
| DSI | values-screened | 4630 | 13.8% | 31.9% | 2.31 | 23% |
| ERTH | environmental | 4654 | 17.7% | 38.2% | 2.16 | 23% |
| ESGD | international ESG | 2178 | 13.5% | 28.5% | 2.12 | 18% |
| esg_index | constructed | 3694 | 11.7% | 23.4% | 2.00 | 35% |
| SUSL | ESG leaders | 1498 | 12.3% | 22.7% | 1.85 | 36% |
| ICLN | clean-energy thematic | 4235 | 19.7% | 33.1% | 1.68 | 46% |

**Every one of the 12 separates volatility states (ratio 1.68–2.92).** The result is a
property of ESG equity markets, not of one basket: it survives values-screened,
low-carbon, fossil-fuel-free, thematic, and international universes.

## Economic value — overlay vs buy & hold (walk-forward, net of costs)

| Index | B&H Sharpe | Overlay Sharpe | ΔSharpe | B&H MaxDD | Overlay MaxDD | DD cut |
|---|---:|---:|---:|---:|---:|---:|
| ESGU | 0.69 | **0.92** | **+0.23** | −33.9% | −14.3% | 58% |
| ESGV | 0.68 | **0.86** | +0.17 | −33.7% | −13.6% | 60% |
| SPYX | 0.74 | 0.88 | +0.14 | −32.8% | −18.3% | 44% |
| SUSA | 0.52 | 0.65 | +0.13 | −53.9% | −18.2% | 66% |
| NULV | 0.53 | 0.60 | +0.07 | −37.0% | −9.3% | **75%** |
| ICLN | 0.08 | 0.12 | +0.04 | −72.5% | −33.5% | 54% |
| SUSL | 0.87 | 0.90 | +0.04 | −27.0% | −8.3% | 69% |
| DSI | 0.56 | 0.56 | −0.00 | −49.9% | −27.8% | 44% |
| CRBN | 0.69 | 0.64 | −0.05 | −33.1% | −14.7% | 56% |
| ERTH | 0.21 | 0.12 | −0.09 | −64.2% | −42.8% | 33% |
| ESGD | 0.42 | 0.20 | −0.23 | −33.7% | −23.1% | 31% |
| esg_index | 0.78 | 0.28 | **−0.51** | −37.2% | −17.4% | 53% |

- **Drawdown protection is universal: 11/11 published indices, average 54% cut.** This
  is the single most robust result in the study.
- Sharpe improves on 7/11. Best on broad ESG (ESGU +0.23, ESGV +0.17, SPYX +0.14).

## Why the constructed basket underperforms — and it's diagnosable

There is a clean monotone relationship between **how often a series is labeled Stress**
and how badly the overlay does:

| Stress % | Indices | Mean ΔSharpe |
|---|---|---:|
| 15–24% (well calibrated) | ESGV, NULV, ESGU, SPYX, SUSA, CRBN, DSI, ERTH, ESGD | +0.04 |
| 35–46% (over-labeling) | esg_index (35%), SUSL (36%), ICLN (46%) | −0.14 |

The constructed basket flags **35% of days as Stress** under walk-forward — roughly
double the well-calibrated indices — so the overlay sits out far too much of a bull
market. Over-labeling also *dilutes* the stress bucket with ordinary days, which is
why its vol ratio (2.00) is below the published mean (2.41).

**Likely cause — a methodology flaw in the constructed index, not in the HMM.** The
basket has no real intraday high/low, so `build_esg_index.py` synthesizes an OHLC bar
from the *cross-sectional* max/min return across 34 constituents. That systematically
inflates the daily range (it is a dispersion measure, not an intraday range), which
inflates the Parkinson range-vol feature and biases the model toward Stress. Note the
same basket under the chronological train/test protocol labeled only 11–13% Stress —
consistent with a feature-scaling artifact that the expanding walk-forward window
amplifies rather than a flaw in the regime concept.

## Label agreement with the constructed basket

| Index | Shared days | Exact match | Stress recall |
|---|---:|---:|---:|
| CRBN | 2600 | 67% | 77% |
| SPYX | 2356 | 63% | 65% |
| ESGV | 1650 | 60% | 58% |
| ESGU | 2100 | 57% | 50% |
| NULV | 2077 | 55% | 45% |
| ESGD | 2171 | 54% | 47% |
| SUSA | 3694 | 53% | 31% |
| DSI | 3694 | 51% | 43% |
| SUSL | 1491 | 50% | 78% |
| ERTH | 3694 | 49% | 43% |
| ICLN | 3694 | 43% | 58% |

Agreement is only **43–67%**, so there is no single universal "ESG regime" — the state
is meaningfully universe-specific. Whichever series the RL allocator trades on should
also be the series its regimes are detected on.

## Recommendation

1. **Use a published broad ESG index (ESGU, ESGV, or SUSA) as the regime substrate.**
   Best vol separation (2.7–2.9), best ΔSharpe (+0.13 to +0.23), and no
   index-construction artifacts. SUSA additionally gives 20 years of history.
2. **Fix or retire the synthetic-OHLC construction** in `build_esg_index.py` — either
   fetch true constituent OHLC and aggregate properly, or drop `parkinson_vol` for the
   constructed basket and rely on close-to-close vol.
3. **Detect regimes on the same universe you allocate over** — cross-index label
   agreement is too low to reuse one regime path everywhere.

Reproduce: `python -m esg_regime.compare_indices`
Raw output: `index_comparison.txt` · data: `index_comparison.csv`, `index_agreement.csv`

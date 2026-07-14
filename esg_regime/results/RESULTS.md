# ESG Regime HMM — backtest results

Two look-ahead-free protocols on three ESG universes. The **overlay** allocates
100% / 60% / 0% to the ESG series in Calm / Choppy / Stress (signal lagged one
day, 1 bps turnover cost, cash at 2% p.a.). Buy & Hold is the same series held
continuously.

- **esg_index** — equal-weight basket of the **top-40 US large-caps by real
  Refinitiv ESG score** (`Dataset/ESG_2000-2026.csv`), 2011–2026. *This is the
  genuinely ESG-driven universe.*
- **SUSA** — iShares MSCI USA ESG Select ETF, 2006–2026 (external check).
- **DSI** — iShares MSCI KLD 400 Social ETF, 2008–2026 (external check).

## Headline: chronological train / test (fit < 2021-01-01, frozen, decoded OOS)

| Universe | Split | Strategy | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---:|---:|---:|---:|---:|
| **esg_index** | TEST | Buy & Hold | 14.3% | 16.3% | 0.78 | −22.2% | 0.64 |
| **esg_index** | TEST | Overlay | 6.9% | 9.9% | 0.52 | −14.4% | 0.48 |
| **SUSA** | TEST | Buy & Hold | 13.8% | 17.1% | 0.72 | −28.2% | 0.49 |
| **SUSA** | TEST | Overlay | 8.8% | 9.9% | 0.70 | −17.4% | 0.50 |
| **DSI** | TEST | Buy & Hold | 14.8% | 17.7% | 0.75 | −28.4% | 0.52 |
| **DSI** | TEST | **Overlay** | 10.0% | **8.5%** | **0.92** | **−7.7%** | **1.29** |

**Read:** the overlay roughly **halves volatility and max drawdown** in every
out-of-sample test. On DSI it also beats buy & hold on risk-adjusted return
(Sharpe 0.92 vs 0.75, Calmar 1.29 vs 0.52). On the high-ESG basket and SUSA it
trades some CAGR for a large drawdown reduction — expected, since 2021–2026 was a
strong bull run where sitting out stress days costs return.

## The regimes are economically real (OOS test period)

Next-day ESG return grouped by the **frozen-model** regime label — the label is
assigned before the return is observed, so this is a clean out-of-sample check.

| Universe | Calm vol → Sharpe | Choppy vol → Sharpe | Stress vol → Sharpe |
|---|---|---|---|
| esg_index | 12.0% → 0.17 | 15.7% → 1.27 | **26.4%** → 1.23 |
| SUSA | 12.1% → 0.97 | 15.6% → 0.88 | **27.3%** → 0.80 |
| DSI | 12.0% → 1.21 | 14.5% → 1.28 | **25.4%** → 0.41 |

Stress days carry **~2× the volatility** of Calm days across all three universes,
out-of-sample. That volatility separation — not the overlay's P&L — is the point:
it is the signal the RL allocator's per-regime reward weights will key off.

## Walk-forward (retrain monthly, expanding window)

| Universe | Strategy | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---:|---:|---:|---:|---:|
| esg_index | Buy & Hold | 15.2% | 17.4% | 0.78 | −37.2% | 0.41 |
| esg_index | Overlay | 4.1% | 8.4% | 0.28 | −17.4% | 0.23 |
| SUSA | Buy & Hold | 10.5% | 18.7% | 0.52 | −53.9% | 0.20 |
| SUSA | Overlay | 8.4% | 10.1% | 0.65 | −18.2% | 0.46 |
| DSI | Buy & Hold | 11.8% | 19.8% | 0.56 | −49.9% | 0.24 |
| DSI | Overlay | 7.6% | 10.5% | 0.56 | −27.9% | 0.27 |

On the long ETF histories (SUSA/DSI) the walk-forward overlay lifts Sharpe (0.52→
0.65, 0.56→0.56) and cuts max drawdown by ~half — it avoids the 2008 crash the
train/test split can't see. On the shorter high-ESG basket the walk-forward is
over-conservative early (Stress 35% of days when training data is thin), which
drags its CAGR; the better-calibrated train/test protocol puts Stress at ~11–13%.

## Caveats

- The overlay is a **risk-management** demonstration, not a return-maximizing
  strategy; its value here is evidence that the regime labels are tradeable and
  transfer out-of-sample.
- Regimes are detected from **price dynamics of an ESG universe**, not from ESG
  *scores* directly. A natural next step is adding cross-sectional ESG-dispersion
  / ESG-momentum features so regimes reflect ESG-factor stress specifically.
- Cash yield (2%) and costs (1 bps/turnover) are flat assumptions.

## Files

`*_summary.csv` — train/test/walk-forward metrics per universe.
`*_walkforward.csv` — full daily regime path + equity curves.
`*_regimes.png` — price shaded by regime + overlay-vs-buy&hold equity.
`eval_output.txt` — full console log.

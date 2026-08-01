# ESG Regime Detection — Full Methodology, Results & Integration

Complete record of the `esg_regime` work: what data was used, how the model was
built and trained, how it was validated, what was found, and how it plugs into
`esg_adaptive_rl`.

---

## 1. Objective

`esg_adaptive_rl` proposes an RL allocator whose reward weighting over return and
the E/S/G factors is **evolved separately per market regime**. That design needs a
regime signal it does not yet have. This package supplies it: a detector that
labels every trading day in the ESG universe as **Calm**, **Choppy**, or **Stress**.

Design constraint: the labels must be **point-in-time** (computable with only past
data), or any downstream RL result inherits look-ahead bias.

---

## 2. Provenance — adapted from a production model

The engine is adapted from an existing, validated SPY regime model
(`TradingAgentV2/v2/regime_core`). Everything structural was kept; only the
feature set changed, because the reference relies on SPX-option-implied data.

| Reference (SPY) | This model (ESG) | Why |
|---|---|---|
| `term_structure = log(vix9d/vix)` | `rv_term_structure = log(rv₁₀/rv₆₀)` | VIX is SPX-specific; realized-vol term structure is computable on any ESG series |
| `vrp` (implied − realized vol) | *dropped* | Requires option-implied vol, unavailable per ESG universe |
| trend, realized_vol, drawdown, parkinson_vol | **kept** | Robust, price-only |
| — | `mom_20` added | 20-day momentum |

Unchanged: 3-state Gaussian HMM, vol-ordered state labeling, filtered posteriors,
anti-whipsaw hysteresis, deterministic stress guardrail, walk-forward protocol.

---

## 3. Data

### 3.1 ESG scores — `Dataset/ESG_2000-2026.csv` (repo-provided)
Refinitiv-style company ESG ratings.

| Property | Value |
|---|---|
| Rows × columns | 3,551 companies × 36 |
| Coverage | US-listed only; ~$83T aggregate market cap |
| Headline field | `ESG Score (FY0)`, 0–100 — mean **40.3**, median 38.3, std 20.1, range 0.8–92.8 |
| History | `ESG Score (FY-1 … FY-25)` annual; coverage thins: 3,551 (FY0) → 845 (FY-10) → 383 (FY-20) → 0 (FY-25) |
| Pillars | Environmental / Social / Governance + Controversies, **FY0 only, on a compressed ~0–5 scale** |
| Identity | RIC, name, country, GICS industry, market cap, ISIN |

**Used for:** selecting the constructed universe only. Not used as a model feature.

### 3.2 Price data — Yahoo Finance (`yfinance`), split/dividend adjusted
Twelve ESG universes, cached to `data/` for reproducibility.

| Universe | Style | History | Bars |
|---|---|---|---:|
| `esg_index` | Score-constructed top-40 ESG basket | 2011–2026 | 3,694 |
| SUSA | Broad ESG (MSCI USA ESG Select) | 2006–2026 | 5,085 |
| DSI | Values-screened (KLD 400 Social) | 2008–2026 | 4,630 |
| ERTH | Environmental / sustainable future | 2008–2026 | 4,654 |
| ICLN | Clean-energy thematic | 2009–2026 | 4,235 |
| CRBN | Low carbon (global ACWI) | 2016–2026 | 2,607 |
| SPYX | Fossil-fuel free (S&P 500) | 2017–2026 | 2,363 |
| ESGD | International ESG (EAFE) | 2017–2026 | 2,178 |
| ESGU | Broad ESG (ESG Aware MSCI USA) | 2018–2026 | 2,107 |
| NULV | ESG large-cap value | 2018–2026 | 2,084 |
| ESGV | Broad ESG (Vanguard) | 2018–2026 | 1,657 |
| SUSL | ESG leaders | 2020–2026 | 1,498 |

---

## 4. Universe construction (`build_esg_index.py`)

1. Read the ESG score table; normalize column names.
2. Filter: `country == "United States of America"` **and** market cap ≥ **$5B**.
3. Rank by `ESG Score (FY0)` descending; take the **top 40**.
   *Note: a global ranking — **not** top-N per industry. It happened to come out
   diversified (max 3 names per GICS industry).*
4. Map RIC → Yahoo ticker (`NVDA.OQ` → `NVDA`).
5. Download adjusted prices from 2010; keep names with >90% history coverage and a
   fully overlapping window → **34 of 40 names survived**.
6. Form an **equal-weight, daily-rebalanced** index: `idx_ret = mean(constituent returns)`,
   level = `100 · cumprod(1 + idx_ret)`.
7. Synthesize an OHLC bar so range-vol estimators work (**see §10.2 — this step is
   the known defect**).

Selected names include JNJ (89.4), Owens Corning (89.0), AngloGold (87.8),
Bank of America (87.6), Microsoft (87.5) — mean score 85.4 vs the 40.3 population mean.

---

## 5. Feature engineering (`features.py`)

Six features, all from price. `close` is adjusted; `logret = log(cₜ/cₜ₋₁)`.

| Feature | Definition |
|---|---|
| `rv_term_structure` | `log(rv₁₀ / rv₆₀)`, where `rvₙ = std(logret, n)·√252`. **>0 ⇒ short-horizon vol elevated (stress/backwardation)** |
| `realized_vol` | `std(logret, 20)·√252` — also used to order the states |
| `trend` | `close / SMA₂₀₀ − 1` |
| `drawdown` | `close / max(close, 252) − 1` |
| `mom_20` | 20-day simple return |
| `parkinson_vol` | `√( mean(log(h/l)², 20) / (4·log2) · 252 )` |

**Standardization (critical).** Each feature is z-scored with a **lagged expanding
window** — mean and std computed through *t−1* only:

```python
m  = s.expanding(min_periods=60).mean().shift(1)
sd = s.expanding(min_periods=60).std().shift(1)
z  = (s - m) / sd
```

The `.shift(1)` is what prevents the scaler from seeing the future. Rows lacking any
core feature are dropped (~300 warm-up bars).

---

## 6. Model & training (`classifier.py`)

**Unsupervised** — there are no regime labels in the data; states emerge from the
feature distribution and are named afterward.

1. **Input:** `X` = the six z-scored features, shape `T × 6`.
2. **Fit:** `GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, tol=1e-3)`,
   trained by **Baum-Welch (EM)**. Learns `startprob_`, the 3×3 `transmat_`, and per-state
   Gaussian emissions (`means_`, `covars_`). Diagonal covariance keeps the parameter
   count low enough to fit stably on short windows.
3. **Restarts:** EM is non-convex → **8 restarts** (`random_state = 0…7`), each scored by
   training log-likelihood; the best model is kept.
4. **State naming:** `predict` on the training data, compute each state's **mean realized
   volatility**, rank them → lowest = `S1_calm`, middle = `S2_choppy`, highest = `S3_stress`.
   Deterministic and interpretable across runs.
5. **Decoding:** the **forward algorithm** gives filtered posteriors
   `P(stateₜ | data₁..ₜ)` — never Viterbi, which would need the whole sequence.
   Label = argmax; confidence = its probability.

**Post-processing (rule-based, not learned):**
- **Hysteresis** — a regime switch is confirmed only at confidence ≥ **0.60** and after
  ≥ **2** consecutive days in the new raw state. Suppresses whipsaw.
- **Guardrail** — force `S3_stress` when `close < SMA₂₀₀` **and** `rv_term_structure > 0`.

---

## 7. Validation protocols

| Protocol | Procedure | Purpose |
|---|---|---|
| **In-sample** | Fit once on full history | Optimistic baseline only |
| **Train/test** | Fit on dates **< 2021-01-01**, **freeze**, forward-decode 2021–2026 | Clean held-out test; no test bar in training |
| **Walk-forward** | Retrain every **21 days** on an expanding window (min 252 days), decode filtered | Every bar labeled only from its own past |

Backtest overlay: **100% / 60% / 0%** invested in Calm / Choppy / Stress; signal
**lagged one day**; cash at 2% p.a.; **1 bps** per unit turnover.

---

## 8. Results — base model

### 8.1 Chronological train/test (out-of-sample, 2021–2026)

| Universe | Strategy | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---:|---:|---:|---:|---:|
| esg_index | Buy & Hold | 14.3% | 16.3% | 0.78 | −22.2% | 0.64 |
| esg_index | Overlay | 6.9% | 9.9% | 0.52 | −14.4% | 0.48 |
| SUSA | Buy & Hold | 13.8% | 17.1% | 0.72 | −28.2% | 0.49 |
| SUSA | Overlay | 8.8% | 9.9% | 0.70 | −17.4% | 0.50 |
| DSI | Buy & Hold | 14.8% | 17.7% | 0.75 | −28.4% | 0.52 |
| DSI | **Overlay** | 10.0% | **8.5%** | **0.92** | **−7.7%** | **1.29** |

### 8.2 Regime-conditional next-day volatility (OOS — the core check)

| Universe | Calm | Choppy | Stress |
|---|---:|---:|---:|
| esg_index | 12.0% | 15.7% | **26.4%** |
| SUSA | 12.1% | 15.6% | **27.3%** |
| DSI | 12.0% | 14.5% | **25.4%** |

Stress days carry ~2× calm-day volatility out-of-sample. The label is assigned
*before* the return is observed, so this is a genuine predictive result.

---

## 9. Results — 12-index comparison (`compare_indices.py`)

Identical pipeline on all 12 universes. Full detail in `INDEX_COMPARISON.md`.

| | Vol ratio (stress/calm) | ΔSharpe | Drawdown cut |
|---|---:|---:|---:|
| Score-constructed basket | 2.00 | **−0.51** | 53% |
| Published ESG indices (11) | **2.41** (range 1.68–2.92) | **+0.04** | 54% |

**Findings**

1. **Regime structure is universal** — all 12 universes separate volatility states
   (ratio 1.68–2.92), across values-screened, low-carbon, fossil-fuel-free, thematic
   and international ESG styles.
2. **Published indices beat the bespoke basket** on both regime quality and
   economics. The ESG-score-constructed universe was **not necessary**.
3. **Drawdown protection is the most robust effect** — 11/11 indices, avg 54% cut.
4. **Best substrates:** ESGU (ΔSharpe +0.23), ESGV (+0.17), SPYX (+0.14), SUSA (+0.13).
5. **No universal ESG regime** — cross-index label agreement is only 43–67%.

---

## 10. Limitations & known defects

### 10.1 The overlay is a demonstration, not a strategy
It trades return for risk in bull markets. Its role is to show the labels are
tradeable and transfer OOS — not to propose an allocation product.

### 10.2 Synthetic OHLC in the constructed basket — **a real defect**
The basket has no true intraday high/low, so `build_esg_index.py` synthesizes a bar
from the **cross-sectional** max/min return across constituents. That is a
*dispersion* measure, not an intraday range: it inflates `parkinson_vol` and biases
the model toward Stress (**35% of days**, vs 15–24% for well-calibrated indices),
which is why that basket has the panel's worst ΔSharpe. Corroborating evidence: the
same basket labeled only 11–13% Stress under train/test, consistent with a
feature-scaling artifact amplified by the expanding window.
**Fix:** aggregate true constituent OHLC, or drop `parkinson_vol` for constructed baskets.

### 10.3 Price regimes, not ESG-score regimes
Regimes are detected from the price dynamics of an ESG universe. ESG scores enter
only through universe selection. Adding cross-sectional ESG dispersion / ESG
momentum as features is the natural next step.

### 10.4 Other
Flat 2% cash and 1 bps costs; ETF proxies carry survivorship/inception bias; the
provider of the ESG scores is inferred, not stated in the repo.

---

## 11. Integration with `esg_adaptive_rl`

`regime.py` deliberately mirrors the repo's conventions (dataclass config,
chronological split, Google-style docstrings) so it drops in cleanly.

### 11.1 Available API
```python
from esg_regime import MarketRegimeDetector, RegimeDetectorConfig, label_regimes

labels = label_regimes(prices, mode="walk_forward")   # date, regime, confidence
# or, explicit train/test control:
det = MarketRegimeDetector(RegimeDetectorConfig(split_date="2021-01-01"))
det.fit(train_features); regimes = det.predict(test_features)
```

### 11.2 Proposed wiring (three steps)

**Step 1 — carry the regime on `MarketData`.** In `data.py`, attach a
`regimes: np.ndarray` of shape `(T,)` aligned to `dates`, produced by
`label_regimes()` on the allocator's own universe. `split_by_date` already subsets
by the same mask, so nothing else changes.

**Step 2 — expose it to the agent in `env.py`.** Append the current regime to the
observation as a 3-dim one-hot. The policy can then condition its allocation on
market state directly.

**Step 3 — make `RewardWeights` regime-indexed.** In `config.py`, replace the single
`DEFAULT_WEIGHTS` with one weight vector per regime:
```python
REGIME_WEIGHTS = {
    "S1_calm":   RewardWeights(w_return=1.0, w_e=0.15, w_s=0.15, w_g=0.15, w_risk=0.3),
    "S2_choppy": RewardWeights(w_return=1.0, w_e=0.10, w_s=0.10, w_g=0.10, w_risk=0.5),
    "S3_stress": RewardWeights(w_return=1.0, w_e=0.05, w_s=0.05, w_g=0.05, w_risk=1.0),
}
```
`reward.py` selects the vector by the current regime. The evolutionary layer then
searches **three** weight vectors instead of one — which is precisely the project's
"evolved separately for each market regime" thesis, and makes the
**regime-conditional price of virtue** directly measurable: the ESG weight the
optimizer tolerates in Calm vs what it abandons in Stress.

### 11.3 Design constraints carried over from the findings
- **Use a broad published ESG index (ESGU/ESGV/SUSA) as the regime substrate**, not
  the constructed basket, until §10.2 is fixed.
- **Detect regimes on the same universe the allocator trades** — 43–67% cross-index
  agreement means a borrowed regime path disagrees roughly half the time.
- Regime labels must come from `walk_forward` (or a frozen train/test model) so the
  RL training set inherits no look-ahead.

---

## 12. File map & reproduction

```
esg_regime/
├── features.py           # point-in-time feature engineering
├── classifier.py         # 3-state Gaussian HMM + hysteresis + guardrail
├── regime.py             # MarketRegimeDetector / RegimeDetectorConfig (integration API)
├── build_esg_index.py    # ESG-score → top-40 universe → equal-weight index
├── evaluate.py           # train/test + walk-forward backtests
├── compare_indices.py    # the 12-index comparison study
├── plot_regimes.py       # regime-shaded price + equity curves
├── data/                 # cached prices, constructed index, universe
├── results/              # metrics CSVs, label paths, plots, RESULTS.md,
│                         # INDEX_COMPARISON.md, METHODOLOGY.md
└── presentation/         # ESG_Regime_HMM.pptx, ESG_Regime_Findings.pptx + generators
```

```bash
pip install -r esg_regime/requirements.txt
python -m esg_regime.build_esg_index     # rebuild the constructed universe
python -m esg_regime.evaluate            # train/test + walk-forward on 3 universes
python -m esg_regime.compare_indices     # the 12-index study
python -m esg_regime.plot_regimes        # figures
```

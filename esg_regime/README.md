# esg_regime — market-regime detection for the ESG universe

A companion package to `esg_adaptive_rl`. It labels each trading day as **Calm
(S1)**, **Choppy (S2)**, or **Stress (S3)** using a 3-state Gaussian hidden
Markov model, so the RL allocator can later condition its per-regime reward
weights (return vs. E/S/G vs. tail risk) on the current market state.

The architecture is adapted from a production SPY regime model. The one
substantive change is the **feature set**: instead of SPX-option-implied VIX term
structure (which only exists for broad indices), it uses a self-contained
*realised-volatility term structure* computed from the ESG universe's own price
history — so it needs no options data and is genuinely ESG-specific.

## What makes it "ESG-specific"

The regime model runs on a **high-ESG price index** built from the repo's real
Refinitiv ESG scores (`Dataset/ESG_2000-2026.csv`): the top-40 US large-caps by
current ESG score are downloaded and formed into an equal-weight daily index
(`build_esg_index.py` → `data/esg_index.csv`). It is also validated on two
third-party ESG ETFs (`SUSA`, `DSI`) as external robustness checks.

## Model

- **Emissions:** 3-state Gaussian HMM, diagonal covariance, 8 random restarts.
- **Features (locked):** `rv_term_structure` (log of 10d/60d realised vol),
  `trend` (close/SMA200−1), `realized_vol`, `drawdown`, `mom_20`, `parkinson_vol`.
  All are standardized with a **lagged expanding z-score** (no look-ahead).
- **Labeling:** states are pinned by mean realised vol (lowest → Calm … highest
  → Stress) so labels are stable across refits.
- **Decoding:** filtered (forward-only) posteriors — labeling day *t* never uses
  data after *t*.
- **Post-processing:** anti-whipsaw hysteresis (confirm a switch only after
  `dwell` days above a confidence threshold) + a deterministic stress guardrail
  (`close < SMA200` **and** short-vol > long-vol → force Stress).

## Validation protocols (both look-ahead-free)

1. **Train/Test** — fit once on dates `< SPLIT_DATE` (2021-01-01), freeze, decode
   the test period out-of-sample with filtered posteriors.
2. **Walk-forward** — retrain monthly on an expanding window; every bar labeled
   by a model trained only on its past.

## Layout

```
esg_regime/
├── features.py          # point-in-time feature engineering
├── classifier.py        # the 3-state Gaussian HMM
├── regime.py            # config + MarketRegimeDetector (matches repo conventions)
├── build_esg_index.py   # builds data/esg_index.csv from the real ESG scores
├── evaluate.py          # train/test + walk-forward backtests
├── plot_regimes.py      # regime-shaded price + equity-curve plots
├── data/                # esg_index.csv, esg_universe.csv, cached ETF prices
└── results/             # per-source *_summary.csv, *_walkforward.csv, *_regimes.png
```

## Reproduce

```bash
pip install -r ../requirements.txt hmmlearn scipy scikit-learn matplotlib
python -m esg_regime.build_esg_index      # build the high-ESG index (needs network)
python -m esg_regime.evaluate             # train/test + walk-forward on all sources
python -m esg_regime.plot_regimes         # write results/*_regimes.png
```

## Programmatic use

```python
from esg_regime import MarketRegimeDetector, RegimeDetectorConfig, split_features_by_date

det = MarketRegimeDetector(RegimeDetectorConfig(split_date="2021-01-01"))
feats = det.compute_features(prices)                 # prices: date,open,high,low,close
train, test = split_features_by_date(feats, "2021-01-01")
det.fit(train)
labels = det.predict(test)                           # out-of-sample regime labels
```

The `regime` column of the returned frame (`S1_calm` / `S2_choppy` / `S3_stress`)
is the hook the evolutionary layer indexes its per-regime reward weights on.

See `results/RESULTS.md` for the latest backtest numbers.

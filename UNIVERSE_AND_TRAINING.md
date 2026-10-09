# Universe Selection & Training Guide

This document specifies (1) the exact protocol used to select the investable universe of
stocks, and (2) a step-by-step sequence the team can follow to load the regime datasets
and run the per-regime training.

---

## 1. Universe selection protocol

**Goal:** a values-compliant, diversified, deep-history opportunity set of ESG-leading
companies, selected by a single clean criterion so it is fully reproducible.

**Data sources**
- **Constituents & sectors:** current S&P 500 membership + GICS sector, scraped from
  Wikipedia (`Dataset/sp500_constituents.csv`).
- **ESG scores:** Refinitiv/LSEG ESG export (`Dataset/ESG_2000-26-ESGC.csv`) — overall
  ESG score per fiscal year (FY0–FY-25) plus E/S/G pillars (FY0–FY-5).
- **Prices/returns:** `yfinance` (split/dividend-adjusted, with `repair=True`).

**The criterion (applied per GICS sector):**
1. Start from **S&P 500** members that have Refinitiv ESG data.
2. Rank each sector's names by **average 2017–2025 overall ESG score**.
3. **Exclude** fossil-fuel, tobacco, and defense companies (next-best ESG name takes
   their place):
   - fossil fuel — GICS `Oil, Gas & Consumable Fuels`, `Energy Equipment & Services`;
   - tobacco — GICS `Tobacco`;
   - defense — GICS `Aerospace & Defense`, **plus** a manual list for defense
     contractors GICS codes elsewhere (`LDOS, RTX, NOC, GD, LHX, HII, TXT, BAH, GE, LMT,
     BA, …`).
4. Keep only names with **pre-2008 price history** (so the backtest reaches the 2008
   crisis; young spin-offs are skipped in favor of the next-best older ESG name).
5. Take the **top 5 per sector**.
6. **Drop the Energy sector entirely** — every S&P 500 Energy name is fossil fuel, so
   there is nothing values-compliant to include. (Clean-energy names sit in Utilities/
   Tech under GICS and were considered, but adding a renewable bucket introduced
   history/definition problems, so Energy is simply excluded.)

**Result: 50 names across 10 sectors (5 each).**

| Sector | Tickers |
|---|---|
| Communication Services | DIS, GOOGL, T, VZ, OMC |
| Consumer Discretionary | CCL, HAS, BBY, F, YUM |
| Consumer Staples | CL, PEP, TGT, HSY, BG |
| Financials | C, SPGI, BAC, STT, JPM |
| Health Care | JNJ, A, GILD, BAX, BDX |
| Industrials | MMM, WM, JCI, CAT, FDX |
| Information Technology | MSFT, INTC, CSCO, FLEX, ACN |
| Materials | CRH, NEM, LIN, IFF, FCX |
| Real Estate | CBRE, HST, VTR, DOC, WY |
| Utilities | PCG, D, XEL, EIX, SRE |

The list lives in `esg_adaptive_rl/config.py` (`UNIVERSE`). Regenerate the supporting
rankings with `python build_top_performers.py` (writes `esg_top_performers.csv`,
`returns_top_performers.csv`, `intersection_universe.csv`).

**Known caveats:** current-S&P-500 membership carries survivorship bias; the E/S/G pillar
data is only fully populated for ~the last 3 years (overall ESG is deeper); Energy has no
representation by design.

> For the backtest, the fixed list above is superseded by the point-in-time,
> year-stamped universe of Section 1.5 — the fixed list's hindsight selection
> (2017–2025 scores traded over 2005–2026) is a look-ahead that the dynamic
> universe removes.

---

## 1.5 Point-in-time universe snapshots (survivorship-free selection)

`build_universe_snapshots.py` re-selects the top-5-per-sector universe **every
year** using only what was known before that year — no hindsight survivors, no
2017–2025 scores traded over 2005. Two artifacts make it work:

- **`Dataset/sp500_membership_history.csv`** — per-ticker membership spans
  (`ticker, sector, industry, entry_date, exit_date`), built by
  `build_membership_history.py` from the *year-end revisions* of Wikipedia's
  "List of S&P 500 companies" (2007–2025). Wikipedia's change-event tables are
  known-incomplete (they omit Bear Stearns, WaMu, GE's removal, ...), so the
  builder samples the full constituent table as it stood at each year-end —
  a complete, as-of membership list per year — and resolves it into spans.
  Spans are annual-granularity: a name present in the year-`Y` table is a
  member from Jan 1 of `Y+1` until Jan 1 of `Y+2` after its last appearance.
  The first year-end table exists for 2007, so **the first snapshot year is
  2008** (the 2005–2007 window stays covered by the fixed legacy universe).
- **`Dataset/universe_snapshots.csv`** — one row per selected name per year:
  `year, ticker, sector, esg_rank, ret_rank, esg_years, ret_years, min_years,
  esg_mean, selected`.

**Selection timing (look-ahead-free, mirrors `esg_data.py`'s publication lag):**
the snapshot for year `Y` is the universe effective Jan 1 of `Y`; membership
comes from the year-end-`(Y-1)` table; an ESG score for calendar year `c` is
known from June of `c+1`, so the ESG leg ranks FY `Y-11 .. Y-2`; annual returns
for `c` are known at year-end, so the return leg ranks calendar years
`Y-10 .. Y-1`. Each leg takes the top-5 per GICS sector per window year; names
top-5 in **both** legs for ≥ 3 window years are "selected", remaining slots are
filled with the best-ESG names (the mandate's exclusion-replacement rule).

**Remaining caveats (all documented in the code):**
- Static current-vintage GICS sectors (the 2016 Real Estate split is not
  restated historically); names whose Refinitiv RIC base ≠ yfinance ticker
  cannot join the ESG table and never rank (same convention as the legacy
  pipeline); entry/exit dates are annual-granularity; the Refinitiv export's
  own historical coverage is thin before ~2006, so early snapshots rank on
  fewer score years.

**Regenerate:**
```bash
python build_membership_history.py   # -> Dataset/sp500_membership_history.csv (needs network)
python build_universe_snapshots.py   # -> Dataset/universe_snapshots.csv (downloads ~950 ticker histories)
```

---

## 2. The regime datasets

Built by `build_regime_datasets.py` and written to `Dataset/regime_datasets/`:

| File | Contents |
|---|---|
| `bull.csv`, `neutral.csv`, `bear.csv` | Long format: `date, ticker, return, esg_E, esg_S, esg_G` — one row per ticker per day, for the days in that regime. |
| `regime_labels.csv` | Daily SPY regime label (`bull`/`neutral`/`bear`). |

**How regimes are defined:** a causal **50/200-day SMA crossover on SPY** with a ±2%
neutral band and a 10-day minimum-dwell filter (no look-ahead — labeling day *t* uses only
data through *t*). `esg_adaptive_rl/regimes.py`.

**Span:** 2005-01-04 → 2026-08-07 (5,432 trading days) · bull 3,762 · neutral 846 ·
bear 824 (the bear window includes the 2008 GFC, 2020, and 2022).

**ESG in the datasets** is the real Refinitiv data, mapped from fiscal years to calendar
days with a reporting lag (look-ahead-safe) and normalized to `[0, 1]`.

> Note on regime returns: the bear regime is **high-volatility and roughly flat**, not
> negative-mean (it is a *trend* definition and includes lagged recoveries). The regimes
> differ mainly by volatility and trajectory.

---

## 3. Running the training — step by step

### Step 0 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 1 — (optional) Rebuild the datasets
Only needed if you change the universe or want to refresh prices (requires network):
```bash
python build_regime_datasets.py      # -> Dataset/regime_datasets/*.csv
```
The split is deterministic, so the regenerated CSVs match the committed ones.

### Step 2 — Run the per-regime evolutionary training
This is the main experiment: for each regime it runs the eight nature-inspired optimizers
(CMA-ES, xNES, DE, L-SHADE, PSO, ACO-R, GWO, ABC) to evolve the reward weighting
`(return, E, S, G, tail-risk)` that performs best on that regime's days.
```bash
python evolve_regimes.py
```
Internally this loads `config.UNIVERSE`, attaches the real ESG, labels SPY regimes, splits
into the three regime datasets, and runs the sweep — printing each optimizer's evolved
weights per regime (the headline "which ESG factor matters in which regime" result).

Tune the run in `evolution/config.py` (`EvolutionConfig`): `population_size`,
`max_generations`, `fitness_timesteps`, `algorithms`, `seed`.

### Loading the cached CSVs directly into the agent
To train on the committed splits **without re-downloading**, load a regime CSV straight
into the environment:
```python
from esg_adaptive_rl.data import load_regime_dataset, split_by_fraction
from esg_adaptive_rl.env import PortfolioEnv
from esg_adaptive_rl.reward import RewardWeights

bear = load_regime_dataset("Dataset/regime_datasets/bear.csv")   # -> MarketData
train, val = split_by_fraction(bear, train_fraction=0.7)         # chronological split
env = PortfolioEnv(train, RewardWeights(w_return=1.0, w_e=0.1, w_s=0.1, w_g=0.1, w_risk=0.5))
# ... train a PPO allocator on `env`, or feed (train, val) to the evolutionary search:
#     from evolution import run_search, EvolutionConfig
#     from evolve_weights import make_objective
#     result = run_search(make_objective(train, val, EvolutionConfig()), EvolutionConfig())
```

### Other entry points
- `python train_single.py` — train one fixed-weight allocator (single regime, sanity check).
- `python evolve_weights.py` — the eight-optimizer sweep on the full period (no regime split).
- `python evolve_meta.py` — **meta-controller backtest**: per-regime specialists (trained
  strictly on dates before the split) switched by the causal label of day `t-1` over the
  test window, with whipsaw (switches/yr) and turnover accounting; runs fully offline from
  the cached regime datasets. See `esg_adaptive_rl/meta.py`.

### Data flow summary
```
config.UNIVERSE ─┐
                 ├─ load_market_data(esg_source="refinitiv")  ─► prices + real ESG
SPY ── load_index_close ─► label_regimes (50/200, causal) ─┐
                                                            ├─ split_by_regime ─► {bull, neutral, bear} MarketData
                                                            │        (cached to Dataset/regime_datasets/*.csv)
                                                            ▼
                                   per regime: PortfolioEnv ─► PPO (inner loop) ─► evolutionary search (outer loop)
                                                            ▼
                                   evolved reward weights per regime  ──►  the finding
```

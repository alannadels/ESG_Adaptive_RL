"""Head-to-head: HMM vs crossover regime labels, graded on the SAME held-out days.

`evolve_regimes_compare.py` scores each labeling inside its own regime buckets,
so its numbers are not comparable across labelings — the buckets contain
different dates. This module runs the architecture the project actually
proposes, which *is* comparable:

    1. Split the timeline chronologically into train / test.
    2. For each labeling, split the TRAIN period into bull / neutral / bear and
       train one PPO specialist allocator per regime.
    3. Walk the TEST period day by day. At each day the labeling's own label
       (point-in-time) selects which specialist acts. This is the meta-controller.
    4. Score the resulting single combined equity curve.

Because every variant is walked over the identical test days, the comparison is
apples-to-apples. Baselines on the same days: one allocator trained on all train
days with no regime switching (isolating what regime-specialisation buys), and
equal-weight buy-and-hold.

Reward weights are held FIXED across all specialists by default, so the only
difference between the two labelings is the labeling itself. Pass --evolved to
use the per-regime evolved weights from `evolve_regimes_compare.py` instead.

Run from the repository root:

    python meta_controller_eval.py
    python meta_controller_eval.py --evolved
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO  # noqa: E402

from esg_adaptive_rl import config as base_config  # noqa: E402
from esg_adaptive_rl.data import MarketData, load_market_data  # noqa: E402
from esg_adaptive_rl.env import PortfolioEnv  # noqa: E402
from esg_adaptive_rl.metrics import (  # noqa: E402
    annualized_return, annualized_volatility, conditional_value_at_risk,
    max_drawdown, sharpe_ratio,
)
from esg_adaptive_rl.reward import RewardWeights  # noqa: E402

REGIMES = ["bull", "neutral", "bear"]
LABEL_FILES = {
    "hmm": "Dataset/regime_datasets_hmm/regime_labels.csv",
    "crossover": "Dataset/regime_datasets/regime_labels.csv",
}
START_DATE, END_DATE = "2005-01-01", "2026-08-09"
ESG_PATH, ESG_ANCHOR_YEAR = "Dataset/ESG_2000-26-ESGC.csv", 2025
SPLIT_DATE = "2019-01-01"          # test window spans COVID-2020 and the 2022 bear
TRAIN_TIMESTEPS = 120_000
SEEDS = (0, 1, 2)                  # specialists retrained per seed; results averaged
OUT_JSON = "esg_regime/results/meta_controller_eval.json"


# --------------------------------------------------------------------- helpers
def subset(data: MarketData, mask: np.ndarray) -> MarketData:
    """Row-subset a MarketData bundle by a boolean day mask."""
    return MarketData(
        dates=data.dates[mask],
        tickers=data.tickers,
        returns=data.returns[mask],
        esg={k: v[mask] for k, v in data.esg.items()},
    )


def train_allocator(data: MarketData, weights: RewardWeights, seed: int) -> PPO:
    """Train one PPO allocator on a MarketData bundle."""
    env = PortfolioEnv(data, weights, lookback=base_config.LOOKBACK,
                       transaction_cost_rate=base_config.TRANSACTION_COST_RATE)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed,
                policy_kwargs={"net_arch": base_config.POLICY_NET_ARCH})
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    return model


def roll_meta(test: MarketData, labels: np.ndarray,
              specialists: Dict[str, PPO], weights: RewardWeights) -> np.ndarray:
    """Walk the test window, letting `labels` pick which specialist acts each day.

    Args:
        test: The held-out MarketData bundle.
        labels: Per-day regime label aligned to ``test.dates``.
        specialists: One trained PPO per regime label.
        weights: Reward weights for the env (affects reward only, not the
            realised returns this function reports).

    Returns:
        The daily net portfolio returns of the switched policy.
    """
    env = PortfolioEnv(test, weights, lookback=base_config.LOOKBACK,
                       transaction_cost_rate=base_config.TRANSACTION_COST_RATE)
    obs, _ = env.reset()
    # Decisions start at index `lookback`; align labels to the same offset.
    t = base_config.LOOKBACK
    done = False
    while not done:
        regime = labels[min(t, len(labels) - 1)]
        model = specialists.get(regime) or next(iter(specialists.values()))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        t += 1
    return env.get_history()["net_returns"]


def score(returns: np.ndarray) -> Dict[str, float]:
    return {
        "CAGR": annualized_return(returns),
        "Vol": annualized_volatility(returns),
        "Sharpe": sharpe_ratio(returns),
        "MaxDD": max_drawdown(returns),
        "CVaR5": conditional_value_at_risk(returns, base_config.CVAR_ALPHA),
    }


def evolved_weights_map(path: str) -> Dict[str, Dict[str, RewardWeights]]:
    """Load per-(labeling, regime) evolved weights, if available."""
    if not os.path.exists(path):
        return {}
    out: Dict[str, Dict[str, RewardWeights]] = {}
    for row in json.load(open(path)):
        if "error" in row:
            continue
        out.setdefault(row["labeling"], {})[row["regime"]] = RewardWeights(*row["weights_raw"])
    return out


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evolved", action="store_true",
                    help="use per-regime evolved weights instead of one fixed vector")
    args = ap.parse_args()

    print(f"Loading universe ({len(base_config.UNIVERSE)} tickers) + real ESG...")
    data = load_market_data(base_config.UNIVERSE, start=START_DATE, end=END_DATE,
                            esg_source="refinitiv", esg_path=ESG_PATH,
                            esg_anchor_year=ESG_ANCHOR_YEAR)
    dates = pd.to_datetime(pd.Index(data.dates))
    is_train = np.asarray(dates < pd.Timestamp(SPLIT_DATE))
    train_all, test = subset(data, is_train), subset(data, ~is_train)
    print(f"Train {train_all.returns.shape[0]} days (< {SPLIT_DATE}) | "
          f"Test {test.returns.shape[0]} days ({pd.Timestamp(dates[~is_train][0]).date()} "
          f"-> {pd.Timestamp(dates[-1]).date()})")

    evolved = evolved_weights_map("esg_regime/results/rl_regime_comparison.json") if args.evolved else {}
    fixed = base_config.DEFAULT_WEIGHTS

    # Per-labeling regime label arrays aligned to the full timeline.
    label_arrays: Dict[str, np.ndarray] = {}
    for name, path in LABEL_FILES.items():
        lab = pd.read_csv(path, parse_dates=["date"]).set_index("date")["regime"]
        label_arrays[name] = lab.reindex(dates, method="ffill").fillna("neutral").to_numpy()

    results: Dict[str, Dict] = {}
    # ---- equal-weight baseline (no model) ----
    ew = test.returns[base_config.LOOKBACK:].mean(axis=1)
    results["equal_weight"] = {"per_seed": [score(ew)], "labels_used": None}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        # ---- no-regime baseline: one allocator on all train days ----
        print("  training single allocator (no regime switching)...")
        single = train_allocator(train_all, fixed, seed)
        r = roll_meta(test, np.array(["_"] * len(dates)), {"_": single}, fixed)
        results.setdefault("no_regime", {"per_seed": []})["per_seed"].append(score(r))

        # ---- one meta-controller per labeling ----
        for labeling, labels in label_arrays.items():
            train_labels = labels[is_train]
            specialists: Dict[str, PPO] = {}
            counts = {}
            for regime in REGIMES:
                mask = train_labels == regime
                counts[regime] = int(mask.sum())
                if mask.sum() <= base_config.LOOKBACK + 5:
                    print(f"  [{labeling}/{regime}] only {mask.sum()} train days; skipped")
                    continue
                w = evolved.get(labeling, {}).get(regime, fixed)
                specialists[regime] = train_allocator(subset(train_all, mask), w, seed)
            print(f"  [{labeling}] trained {len(specialists)} specialists on {counts}")
            r = roll_meta(test, labels[~is_train], specialists, fixed)
            key = f"meta_{labeling}"
            results.setdefault(key, {"per_seed": [], "train_day_counts": counts})
            results[key]["per_seed"].append(score(r))

    # ---- report ----
    def avg(rows: List[Dict[str, float]], k: str) -> float:
        return float(np.mean([r[k] for r in rows]))

    print("\n" + "=" * 96)
    print(f"  HELD-OUT TEST ({SPLIT_DATE} onward) — identical days for every variant, "
          f"mean of {len(SEEDS)} seeds")
    print("=" * 96)
    print(f"{'variant':<22}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'MaxDD':>10}{'CVaR5':>9}")
    order = ["meta_hmm", "meta_crossover", "no_regime", "equal_weight"]
    for key in order:
        if key not in results:
            continue
        rows = results[key]["per_seed"]
        print(f"{key:<22}{avg(rows,'CAGR')*100:>8.2f}%{avg(rows,'Vol')*100:>8.2f}%"
              f"{avg(rows,'Sharpe'):>9.3f}{avg(rows,'MaxDD')*100:>9.2f}%"
              f"{avg(rows,'CVaR5')*100:>8.2f}%")

    if "meta_hmm" in results and "meta_crossover" in results:
        h = [r["Sharpe"] for r in results["meta_hmm"]["per_seed"]]
        c = [r["Sharpe"] for r in results["meta_crossover"]["per_seed"]]
        print(f"\n  Sharpe by seed — hmm {['%.3f' % x for x in h]} | "
              f"crossover {['%.3f' % x for x in c]}")
        print(f"  Mean difference (hmm - crossover): {np.mean(h) - np.mean(c):+.3f}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump({"split_date": SPLIT_DATE, "seeds": list(SEEDS),
               "train_timesteps": TRAIN_TIMESTEPS,
               "weights": "evolved" if args.evolved else "fixed_default",
               "results": results}, open(OUT_JSON, "w"), indent=2, default=float)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()

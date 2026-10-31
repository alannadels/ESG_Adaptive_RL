"""Post-run portfolio data saver -- makes a sweep cell analysable.

The sweep worker saves only the evolved reward weights + best fitness per job. This script
turns one cell's results into full portfolio-performance artifacts for post-run analysis:
for every (regime, optimizer) it re-trains the PPO allocator under the evolved reward
(mirroring evolution.fitness.evaluate_weights), rolls it out on the regime's validation
window across several seeds, and saves the complete picture -- plus the 1/N and SPY
baselines on the identical windows.

Writes, under ``--out``:
  evolved_metrics.csv    one row per (regime, optimizer, seed): sharpe, sortino, calmar,
                         max_drawdown, cvar, annual return/vol, turnover, ESG exposure,
                         and SPY-relative beta/alpha.
  evolved_returns.parquet  long (regime, optimizer, seed, date, net_return) daily series
                         -- enough to rebuild equity curves, drawdowns, rolling metrics.
  evolved_holdings.csv   mean allocation to each ticker per (regime, optimizer).
  baselines_metrics.csv  1/N (equal-weight over tradable names) and SPY, per regime.
  baselines_returns.parquet  daily series for the 1/N and SPY baselines.

    python save_portfolio_data.py --results-dir <cell> --data-cache <pickle> \\
        --timesteps 100000 --seeds 3 --out <cell>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import load_index_close, split_by_fraction
from esg_adaptive_rl.env import PortfolioEnv
from esg_adaptive_rl.metrics import summarize
from esg_adaptive_rl.reward import RewardWeights

TD = 252
REGIMES = ("bull", "neutral", "bear")


def _spy_daily(dates: pd.DatetimeIndex) -> np.ndarray:
    """SPY daily returns aligned to ``dates`` (for beta/alpha and the SPY baseline)."""
    px = load_index_close("SPY", base_config.START_DATE, base_config.END_DATE)
    r = pd.Series(px).astype(float).pct_change()
    r.index = pd.to_datetime(r.index)
    return r.reindex(pd.to_datetime(dates)).to_numpy()


def _beta_alpha(port: np.ndarray, bench: np.ndarray) -> tuple[float, float]:
    """CAPM beta and annualized Jensen alpha of ``port`` vs ``bench`` (NaN-safe)."""
    m = np.isfinite(port) & np.isfinite(bench)
    if m.sum() < 3 or np.var(bench[m]) == 0:
        return float("nan"), float("nan")
    beta = float(np.cov(port[m], bench[m])[0, 1] / np.var(bench[m]))
    alpha = float((port[m].mean() - beta * bench[m].mean()) * TD)
    return beta, alpha


def _rollout(weights: RewardWeights, train, val, timesteps: int, seed: int):
    """Train PPO under ``weights`` on ``train``, roll out on ``val``; return (history, holdings)."""
    train_env = PortfolioEnv(data=train, reward_weights=weights, lookback=base_config.LOOKBACK,
                             transaction_cost_rate=base_config.TRANSACTION_COST_RATE)
    model = PPO(policy="MlpPolicy", env=train_env,
                policy_kwargs={"net_arch": base_config.POLICY_NET_ARCH}, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps)
    env = PortfolioEnv(data=val, reward_weights=weights, lookback=base_config.LOOKBACK,
                       transaction_cost_rate=base_config.TRANSACTION_COST_RATE)
    obs, _ = env.reset(seed=seed)
    held = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, _info = env.step(action)
        held.append(env._prev_weights.copy())   # applied (tradability-masked) weights this day
        done = term or trunc
    return env.get_history(), np.asarray(held)


def _sortino(net: np.ndarray) -> float:
    """Annualized Sortino (downside-deviation) ratio from a daily return series."""
    x = net[np.isfinite(net)]
    downside = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2)) if len(x) else 0.0
    return float(np.sqrt(TD) * x.mean() / downside) if downside > 0 else float("nan")


def _metric_row(reg, opt, seed, hist, spy):
    """Assemble one metrics row from a rollout history + SPY alignment.

    ``sortino`` and ``calmar`` are computed here because the project's ``summarize`` does
    not return them (it exposes sharpe / drawdown / cvar / annual return+vol / ESG / turnover).
    """
    m = summarize(hist, alpha=base_config.CVAR_ALPHA)
    net = np.asarray(hist["net_returns"], dtype=float)
    beta, alpha = _beta_alpha(net, spy[-len(net):] if spy is not None else np.full(len(net), np.nan))
    mdd = m["max_drawdown"]
    calmar = float(m["annual_return"] / abs(mdd)) if mdd not in (0, None) and np.isfinite(mdd) and mdd != 0 else float("nan")
    return {"regime": reg, "optimizer": opt, "seed": seed,
            "sharpe": m["sharpe"], "sortino": _sortino(net), "calmar": calmar,
            "max_drawdown": mdd, "cvar_5": m.get("cvar_5"),
            "annual_return": m["annual_return"], "annual_volatility": m["annual_volatility"],
            "turnover": m["avg_turnover"], "esg_E": m["avg_esg_E"], "esg_S": m["avg_esg_S"],
            "esg_G": m["avg_esg_G"], "beta_SPY": beta, "alpha_SPY": alpha}


def _equal_weight(val) -> np.ndarray:
    """Daily 1/N returns over the tradable names each day (mirrors the mask the agent sees)."""
    r, tr = val.returns, val.tradable
    out = np.empty(r.shape[0])
    for t in range(r.shape[0]):
        m = tr[t] if tr is not None else np.ones(r.shape[1], dtype=bool)
        out[t] = r[t, m].mean() if m.any() else 0.0
    return out[base_config.LOOKBACK:]   # align to the agent's eval trajectory


def main() -> None:
    """Re-evaluate every evolved weight vector in a cell and save full portfolio artifacts."""
    ap = argparse.ArgumentParser(description="Save portfolio-performance data for one sweep cell.")
    ap.add_argument("--results-dir", required=True, help="Dir of the cell's sweep JSONs.")
    ap.add_argument("--data-cache", required=True, help="The regime-split pickle used for the cell.")
    ap.add_argument("--timesteps", type=int, required=True, help="PPO eval budget (match the cell).")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.data_cache, "rb") as fh:
        splits = pickle.load(fh)
    records = []
    for p in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        with open(p) as fh:
            records.append(json.load(fh))
    print(f"[{args.out}] {len(records)} evolved weight vectors; {args.seeds} seeds, "
          f"{args.timesteps} steps", flush=True)

    met_rows, ret_rows, hold_rows = [], [], []
    bmet_rows, bret_rows = [], []
    seeds = [42 + i for i in range(args.seeds)]

    for reg in REGIMES:
        if reg not in splits:
            continue
        train, val = split_by_fraction(splits[reg], train_fraction=0.7)
        dates = pd.to_datetime(val.dates)[base_config.LOOKBACK:]
        spy = _spy_daily(dates)

        # --- baselines on the same val window ---
        ew = _equal_weight(val)
        for strat, series in (("equal_weight", ew), ("SPY", spy)):
            s = np.asarray(series, dtype=float)
            x = s[np.isfinite(s)]
            shp = float(np.sqrt(TD) * x.mean() / x.std()) if len(x) > 1 and x.std() > 0 else float("nan")
            beta, alpha = _beta_alpha(s, spy)
            bmet_rows.append({"regime": reg, "strategy": strat, "sharpe": shp,
                              "annual_return": float((1 + x).prod() ** (TD / max(len(x), 1)) - 1),
                              "beta_SPY": beta, "alpha_SPY": alpha})
            for d, v in zip(dates, s):
                bret_rows.append({"regime": reg, "strategy": strat, "date": d, "net_return": v})

        # --- evolved: one rollout per (optimizer, seed) ---
        for rec in records:
            if rec.get("regime") != reg:
                continue
            opt = rec["algorithm"]
            w = RewardWeights(*np.asarray(rec["best_weights_raw"], dtype=float))
            seed_holdings = []
            for seed in seeds:
                try:
                    hist, held = _rollout(w, train, val, args.timesteps, seed)
                except Exception as exc:  # noqa: BLE001 - a bad rollout shouldn't abort the cell
                    print(f"  WARN {reg}/{opt}/seed{seed} rollout failed: {exc}", flush=True)
                    continue
                met_rows.append(_metric_row(reg, opt, seed, hist, spy))
                net = np.asarray(hist["net_returns"], dtype=float)
                for d, v in zip(dates[-len(net):], net):
                    ret_rows.append({"regime": reg, "optimizer": opt, "seed": seed,
                                     "date": d, "net_return": float(v)})
                if held.size:
                    seed_holdings.append(held.mean(axis=0))
            if seed_holdings:
                mean_hold = np.mean(seed_holdings, axis=0)
                for tk, wt in zip(splits[reg].tickers, mean_hold):
                    hold_rows.append({"regime": reg, "optimizer": opt, "ticker": tk,
                                      "mean_weight": float(wt)})
        print(f"  {reg}: done", flush=True)

    pd.DataFrame(met_rows).to_csv(os.path.join(args.out, "evolved_metrics.csv"), index=False)
    pd.DataFrame(ret_rows).to_parquet(os.path.join(args.out, "evolved_returns.parquet"), index=False)
    pd.DataFrame(hold_rows).to_csv(os.path.join(args.out, "evolved_holdings.csv"), index=False)
    pd.DataFrame(bmet_rows).to_csv(os.path.join(args.out, "baselines_metrics.csv"), index=False)
    pd.DataFrame(bret_rows).to_parquet(os.path.join(args.out, "baselines_returns.parquet"), index=False)
    print(f"[{args.out}] wrote evolved_metrics/returns/holdings + baselines_metrics/returns", flush=True)


if __name__ == "__main__":
    main()

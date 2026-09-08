"""Entry point: regime-switching meta-controller backtest across specialists.

Closes the loop between the regime layer and the RL layer. For each regime
labeling (walk-forward HMM and the causal 50/200 crossover) it:

    1. reconstructs the FULL dataset offline from the cached long-format regime
       CSVs (``Dataset/regime_datasets*``) — no network needed;
    2. splits chronologically at ``--split-date`` (default 2021-01-01): the
       specialists and the single-policy baselines are trained ONLY on dates
       before the split, and everything is evaluated ONLY on the window at or
       after it (strictly look-ahead-free);
    3. trains the three regime specialists per labeling:
         - ``evolved``   — each regime's PPO is trained under that regime's
           evolved reward weights (loaded from the committed
           ``esg_regime/results/rl_regime_comparison.json``);
         - ``default``   — each regime's PPO is trained under the project's
           fixed ``DEFAULT_WEIGHTS`` (isolates the effect of switching with a
           common weighting);
    4. trains two single-policy baselines on the full training window:
       ``DEFAULT_WEIGHTS`` and return-only;
    5. rolls the switched strategy (:class:`esg_adaptive_rl.meta.MetaController`)
       and the single policies deterministically through the test window, net
       of transaction costs, and reports Sharpe/vol/MaxDD/CAGR/CVaR, ESG
       profile, turnover, switches per year, and per-regime exposure — plus an
       equal-weight buy & hold for reference.

The switching signal is the causal label of day ``t-1`` (same as-of convention
as the regime datasets and the ``esg_regime`` overlays), so label timing never
uses same-day information. If a regime has too few training days to fit a
specialist, the controller falls back to the full-window policy for that label
and the fallback is reported.

Honest caveats (same as RL_REGIME_TRAINING.md): single seed, reduced PPO
budget; the evolved weights are the CMA-ES discoveries from the comparison
run, not a fresh full-cost search per meta run. Treat every number here as
directional until the multi-seed generalization pass.

Run from the repository root:

    python evolve_meta.py                        # both labelings, 20k PPO steps/specialist
    python evolve_meta.py --timesteps 100000     # higher PPO budget per specialist
    python evolve_meta.py --labeling hmm         # one labeling only

Outputs: esg_regime/results/meta_controller.json (+ meta_controller.txt log).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import MarketData, split_by_date
from esg_adaptive_rl.env import PortfolioEnv
from esg_adaptive_rl.meta import MetaController
from esg_adaptive_rl.metrics import summarize
from esg_adaptive_rl.reward import RewardWeights
from esg_adaptive_rl.regimes import REGIMES, split_by_regime

REGIME_DIRS: Dict[str, str] = {
    "hmm": "Dataset/regime_datasets_hmm",     # walk-forward 3-state Gaussian HMM
    "crossover": "Dataset/regime_datasets",   # causal 50/200 SMA crossover
}
WEIGHTS_JSON = "esg_regime/results/rl_regime_comparison.json"

# Minimum usable training days for a regime specialist (lookback + margin).
_MIN_TRAIN_DAYS = base_config.LOOKBACK + 20

REPORT_COLS: List[Tuple[str, str]] = [
    ("sharpe", "sharpe"),
    ("annual_volatility", "vol"),
    ("max_drawdown", "maxdd"),
    ("annual_return", "cagr"),
    ("cvar_5", "cvar05"),
    ("avg_turnover", "turn"),
    ("avg_esg_E", "E"),
    ("avg_esg_S", "S"),
    ("avg_esg_G", "G"),
]


# ---------------------------------------------------------------------- data
def load_full_dataset(regime_dir: str) -> MarketData:
    """Reconstruct the full dataset from the three cached long-format CSVs.

    Each regime CSV is one row per (date, ticker); the union of the three
    regimes covers every trading day of the original download, so pivoting back
    to wide returns + E/S/G reproduces the full :class:`MarketData` offline.

    Args:
        regime_dir: Directory holding ``bull.csv``, ``neutral.csv``,
            ``bear.csv`` (and ``regime_labels.csv``).

    Returns:
        A :class:`~esg_adaptive_rl.data.MarketData` with dates in ascending
        order and tickers in ``config.UNIVERSE`` order (missing tickers
        dropped with a warning).
    """
    frames = [
        pd.read_csv(os.path.join(regime_dir, f"{regime}.csv"), parse_dates=["date"])
        for regime in REGIMES
    ]
    df = pd.concat(frames, ignore_index=True)

    raw_returns = df.pivot(index="date", columns="ticker", values="return")

    # Keep universe tickers that survived the pivot; warn about any absent ones.
    present = [t for t in base_config.UNIVERSE if t in raw_returns.columns]
    missing = [t for t in base_config.UNIVERSE if t not in raw_returns.columns]
    if missing:
        print(f"  WARNING: tickers missing from cached datasets: {missing}")
    if not present:
        raise ValueError("no universe tickers found in the cached regime datasets.")

    returns = raw_returns[present].dropna(how="any")
    esg = {
        pillar: (
            df.pivot(index="date", columns="ticker", values=f"esg_{pillar}")
            .reindex(index=returns.index, columns=present)
            .to_numpy(dtype=np.float64)
        )
        for pillar in ("E", "S", "G")
    }
    return MarketData(
        dates=returns.index,
        tickers=list(present),
        returns=returns.to_numpy(dtype=np.float64),
        esg=esg,
    )


def load_labels(regime_dir: str) -> pd.Series:
    """Load the cached day-by-day regime path for one labeling.

    Args:
        regime_dir: Directory holding ``regime_labels.csv``.

    Returns:
        A ``pd.Series`` of bull/neutral/bear labels indexed by date (sorted).

    Raises:
        ValueError: If the labels contain values outside the regime vocabulary.
    """
    df = pd.read_csv(os.path.join(regime_dir, "regime_labels.csv"), parse_dates=["date"])
    labels = pd.Series(df["regime"].values, index=pd.to_datetime(df["date"]))
    bad = set(labels.unique()) - set(REGIMES)
    if bad:
        raise ValueError(f"unexpected regime labels in {regime_dir}: {sorted(bad)}")
    return labels.sort_index()


def load_evolved_weights(path: str) -> Dict[str, Dict[str, RewardWeights]]:
    """Load the per-regime evolved reward weights from the comparison JSON.

    Args:
        path: Path to ``rl_regime_comparison.json``.

    Returns:
        Mapping ``{labeling: {regime: RewardWeights}}`` using the *raw*
        (non-normalized) discovered vectors, exactly as re-scored in the
        comparison run.

    Raises:
        RuntimeError: If the file is missing or does not cover every
            (labeling, regime) cell.
    """
    if not os.path.exists(path):
        raise RuntimeError(
            f"{path} not found — run `python evolve_regimes_compare.py` first "
            "(or pass --weights-json)."
        )
    with open(path) as fh:
        rows = json.load(fh)

    out: Dict[str, Dict[str, RewardWeights]] = {
        labeling: {} for labeling in REGIME_DIRS
    }
    for row in rows:
        labeling, regime = row["labeling"], row["regime"]
        if labeling in out and regime in REGIMES:
            out[labeling][regime] = RewardWeights(*row["weights_raw"])
    for labeling, weights in out.items():
        missing = [r for r in REGIMES if r not in weights]
        if missing:
            raise RuntimeError(
                f"{path} lacks evolved weights for {labeling}/{missing}."
            )
    return out


# ----------------------------------------------------------------- training
def train_policy(
    train_data: MarketData,
    weights: RewardWeights,
    timesteps: int,
    seed: int,
) -> Optional[PPO]:
    """Train one PPO allocator under ``weights`` on ``train_data``.

    Args:
        train_data: Training window (a regime subset or the full window).
        weights: Reward weighting the allocator is trained under.
        timesteps: PPO training budget (environment steps).
        seed: RNG seed so candidates/regimes are compared on equal footing.

    Returns:
        The trained model, or ``None`` if the window is too short or training
        fails (the caller then falls back to the full-window policy).
    """
    if train_data.returns.shape[0] <= _MIN_TRAIN_DAYS:
        return None
    try:
        env = PortfolioEnv(
            data=train_data,
            reward_weights=weights,
            lookback=base_config.LOOKBACK,
            transaction_cost_rate=base_config.TRANSACTION_COST_RATE,
        )
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs={"net_arch": base_config.POLICY_NET_ARCH},
            seed=seed,
            verbose=0,
        )
        model.learn(total_timesteps=timesteps)
        return model
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  WARNING: training failed ({exc}); will fall back.")
        return None


def roll_policy(model: PPO, data: MarketData, weights: RewardWeights, seed: int) -> dict:
    """Roll a trained allocator deterministically through a held-out window.

    Args:
        model: A trained PPO allocator.
        data: The evaluation window.
        weights: The reward weighting (used by the evaluation env; the metrics
            are computed from the recorded trajectory).
        seed: Environment RNG seed.

    Returns:
        A :func:`~esg_adaptive_rl.metrics.summarize` metrics dict.
    """
    env = PortfolioEnv(
        data=data,
        reward_weights=weights,
        lookback=base_config.LOOKBACK,
        transaction_cost_rate=base_config.TRANSACTION_COST_RATE,
    )
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
    return summarize(env.get_history(), alpha=base_config.CVAR_ALPHA)


def equal_weight_metrics(data: MarketData) -> dict:
    """1/N buy & hold over the window (no costs, no turnover)."""
    n = data.returns.shape[0]
    daily = data.returns.mean(axis=1)
    return summarize(
        {
            "net_returns": daily,
            "gross_returns": daily,
            "turnover": np.zeros(n),
            "esg_E": data.esg["E"].mean(axis=1),
            "esg_S": data.esg["S"].mean(axis=1),
            "esg_G": data.esg["G"].mean(axis=1),
        },
        alpha=base_config.CVAR_ALPHA,
    )


# ---------------------------------------------------------------- reporting
def format_number(value: float, key: str) -> str:
    """Format one metric per column style (percent vs plain)."""
    if key in ("annual_volatility", "max_drawdown", "annual_return"):
        return f"{value:>8.2%}"
    return f"{value:>8.3f}"


def format_strategy_row(
    name: str,
    metrics: dict,
    switches_per_year: Optional[float] = None,
) -> str:
    """One report row: strategy name, metrics, optional whipsaw column."""
    cells = "".join(format_number(metrics[key], key) for key, _ in REPORT_COLS)
    switch_cell = (
        f"{switches_per_year:>8.2f}" if switches_per_year is not None else " " * 8
    )
    return f"{name:<17}{cells}{switch_cell}"


def format_report(results: List[dict]) -> str:
    """Human-readable report over the per-labeling result records."""
    header = (
        f"{'strategy':<17}"
        + "".join(f"{alias:>8}" for _, alias in REPORT_COLS)
        + f"{'sw/yr':>8}"
    )
    lines: List[str] = []
    for row in results:
        lines.append("")
        lines.append(
            f"=== labeling={row['labeling']}  split={row['split_date']}  "
            f"train={row['days_train']}  test={row['days_test']}  "
            f"timesteps={row['timesteps']}  seed={row['seed']} ==="
        )
        day_counts = ", ".join(
            f"{regime}={count}" for regime, count in row["regime_train_days"].items()
        )
        lines.append(f"regime train days : {day_counts}")
        fell = row["fell_back"] or "none"
        lines.append(f"specialists evolved: {', '.join(row['specialists']) or 'none'}   "
                     f"fell back: {fell}")
        lines.append("")
        lines.append(header)
        lines.append("-" * len(header))
        for name, metrics in row["strategies"].items():
            if name.startswith("meta_"):
                lines.append(
                    format_strategy_row(name, metrics, metrics["switches_per_year"])
                )
                exposure = ", ".join(
                    f"{r}={pct}%" for r, pct in metrics["regime_exposure"].items()
                )
                lines.append(f"{'exposure':<17}{'':>16}{exposure}")
            else:
                lines.append(format_strategy_row(name, metrics))
        lines.append("-" * len(header))
        meta = row["strategies"]["meta_evolved"]
        single = row["strategies"]["single_default"]
        lines.append(
            f"read: meta_evolved sharpe={meta['sharpe']:.3f} vs "
            f"single_default={single['sharpe']:.3f} "
            f"(delta {meta['sharpe'] - single['sharpe']:+.3f}), "
            f"switches/yr={meta['switches_per_year']:.2f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------- main
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-regime specialists + meta-controller backtest."
    )
    parser.add_argument(
        "--labeling",
        choices=list(REGIME_DIRS),
        default=None,
        help="Run only this regime labeling (default: both).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="PPO training budget per specialist/baseline (default 20000).",
    )
    parser.add_argument(
        "--split-date",
        default="2021-01-01",
        help="Chronological boundary: train < date, evaluate >= date.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--weights-json",
        default=WEIGHTS_JSON,
        help="Path to rl_regime_comparison.json with the evolved weights.",
    )
    parser.add_argument(
        "--out",
        default="esg_regime/results/meta_controller",
        help="Output path prefix (writes .json and .txt).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the meta-controller backtest for the requested labelings."""
    args = _parse_args()
    np.random.seed(args.seed)
    weights_by_labeling = load_evolved_weights(args.weights_json)
    labelings = [args.labeling] if args.labeling else list(REGIME_DIRS)

    print(
        f"Meta-controller backtest (split {args.split_date}, "
        f"{args.timesteps} PPO steps per policy, seed {args.seed})."
    )
    results: List[dict] = []
    for labeling in labelings:
        regime_dir = REGIME_DIRS[labeling]
        t0 = time.time()
        print(f"\n--- {labeling} ---")
        data = load_full_dataset(regime_dir)
        labels = load_labels(regime_dir)
        train_data, test_data = split_by_date(data, args.split_date)
        print(
            f"full={data.returns.shape[0]} days, train={train_data.returns.shape[0]}, "
            f"test={test_data.returns.shape[0]}"
        )

        # Per-regime training subsets, strictly before the split.
        subsets = split_by_regime(train_data, labels)
        evolved_weights = weights_by_labeling[labeling]

        spec_evolved: Dict[str, PPO] = {}
        spec_default: Dict[str, PPO] = {}
        fell_back: List[str] = []
        for regime in REGIMES:
            subset = subsets.get(regime)
            n_days = 0 if subset is None else subset.returns.shape[0]
            print(
                f"  {regime:<8} train days {n_days:>5} "
                f"{'(fall back)' if n_days <= _MIN_TRAIN_DAYS else ''}"
            )
            if n_days <= _MIN_TRAIN_DAYS:
                fell_back.append(regime)
                continue

            evolved_model = train_policy(
                subset, evolved_weights[regime], args.timesteps, args.seed
            )
            if evolved_model is not None:
                spec_evolved[regime] = evolved_model
            else:
                fell_back.append(regime)

            default_model = train_policy(
                subset, base_config.DEFAULT_WEIGHTS, args.timesteps, args.seed
            )
            if default_model is not None:
                spec_default[regime] = default_model
            else:
                fell_back.append(regime)

        # Single-policy baselines on the full training window.
        print("  training single-policy baselines (default, return-only)...")
        single_default = train_policy(
            train_data, base_config.DEFAULT_WEIGHTS, args.timesteps, args.seed
        )
        single_return = train_policy(
            train_data, RewardWeights(w_return=1.0), args.timesteps, args.seed
        )
        if single_default is None:
            raise RuntimeError("full-window default-weight policy failed to train.")

        # Evaluate everything on the same test window.
        print("  evaluating on test window...")
        meta_evolved = MetaController(
            policies=spec_evolved,
            labels=labels,
            default_policy=single_default,
            lookback=base_config.LOOKBACK,
            transaction_cost_rate=base_config.TRANSACTION_COST_RATE,
        ).evaluate(test_data, seed=args.seed)
        meta_default = MetaController(
            policies=spec_default,
            labels=labels,
            default_policy=single_default,
            lookback=base_config.LOOKBACK,
            transaction_cost_rate=base_config.TRANSACTION_COST_RATE,
        ).evaluate(test_data, seed=args.seed)
        single_default_m = roll_policy(
            single_default, test_data, base_config.DEFAULT_WEIGHTS, args.seed
        )
        single_return_m = roll_policy(
            single_return, test_data, RewardWeights(w_return=1.0), args.seed
        )
        bh_m = equal_weight_metrics(test_data)

        results.append(
            {
                "labeling": labeling,
                "split_date": args.split_date,
                "days_train": int(train_data.returns.shape[0]),
                "days_test": int(test_data.returns.shape[0]),
                "regime_train_days": {
                    regime: int(subset.returns.shape[0])
                    for regime, subset in subsets.items()
                },
                "specialists": sorted(spec_evolved),
                "fell_back": fell_back,
                "timesteps": args.timesteps,
                "seed": args.seed,
                "strategies": {
                    "meta_evolved": meta_evolved,
                    "meta_default": meta_default,
                    "single_default": single_default_m,
                    "single_return_only": single_return_m,
                    "equal_weight": bh_m,
                },
                "elapsed_s": round(time.time() - t0, 1),
            }
        )

    report = format_report(results)
    print(report)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(f"{args.out}.json", "w") as fh:
        json.dump(results, fh, indent=2)
    with open(f"{args.out}.txt", "w") as fh:
        fh.write(report + "\n")
    print(f"\nSaved {args.out}.json and {args.out}.txt")


if __name__ == "__main__":
    main()

"""Central configuration for the v0 single-regime pipeline.

All tunable parameters live here so a run can be reproduced and modified from one place.
As the project grows (evolutionary search, market regimes, the meta-controller), this is
where the corresponding settings will be added.
"""

from __future__ import annotations

from typing import List

from esg_adaptive_rl.reward import RewardWeights

# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------
# A small, deliberately mixed set of large, liquid names spanning clean-energy,
# traditional-energy, technology, financial, consumer, and industrial sectors. This is a
# placeholder universe for the backbone; it is meant only to exercise the pipeline and
# can be replaced freely.
UNIVERSE: List[str] = [
    "NEE", "ENPH", "FSLR",   # renewables / clean energy
    "XOM", "CVX", "COP",     # traditional energy
    "AAPL", "MSFT", "NVDA",  # technology
    "JPM", "BAC",            # financials
    "PG", "KO",              # consumer staples
    "TSLA", "GE",            # auto / industrial
]

# --------------------------------------------------------------------------------------
# Date ranges (chronological train/test split to avoid look-ahead)
# --------------------------------------------------------------------------------------
START_DATE: str = "2015-01-01"
END_DATE: str = "2023-12-31"
SPLIT_DATE: str = "2021-01-01"  # train: dates < this; test: dates >= this

# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------
LOOKBACK: int = 20                     # trailing window (days) for return statistics
TRANSACTION_COST_RATE: float = 0.0010  # 10 bps charged per unit of turnover

# The single fixed reward trade-off used by the v0 backbone. Profit-led, with light ESG
# tilts and a moderate downside penalty. Later, these weights become what the
# evolutionary loop searches and what the regime layer swaps per regime.
DEFAULT_WEIGHTS: RewardWeights = RewardWeights(
    w_return=1.0,
    w_e=0.1,
    w_s=0.1,
    w_g=0.1,
    w_risk=0.5,
)

# --------------------------------------------------------------------------------------
# PPO training
# --------------------------------------------------------------------------------------
TOTAL_TIMESTEPS: int = 100_000   # environment steps of training
POLICY_NET_ARCH: List[int] = [128, 128]  # MLP hidden layers for actor and critic
SEED: int = 42

# Tail level for the CVaR evaluation metric (worst 5% of days).
CVAR_ALPHA: float = 0.05

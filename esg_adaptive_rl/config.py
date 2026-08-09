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
# The values-screened investable universe, by one clean criterion: the top-5 ESG
# performers in each of the 10 non-Energy GICS sectors (S&P 500, ranked by average
# 2017-2025 Refinitiv ESG), skipping fossil-fuel, tobacco, and defense companies (the
# next-best ESG name takes their place, including defense contractors GICS codes outside
# "Aerospace & Defense", e.g. LDOS). Restricted to names with pre-2008 price history so
# the backtest reaches the 2008 crisis. The Energy sector is dropped entirely (it is all
# fossil fuel). 50 names across 10 sectors (5 each).
UNIVERSE: List[str] = [
    "DIS", "GOOGL", "T", "VZ", "OMC",       # Communication Services
    "CCL", "HAS", "BBY", "F", "YUM",        # Consumer Discretionary
    "CL", "PEP", "TGT", "HSY", "BG",        # Consumer Staples
    "C", "SPGI", "BAC", "STT", "JPM",       # Financials
    "JNJ", "A", "GILD", "BAX", "BDX",       # Health Care
    "MMM", "WM", "JCI", "CAT", "FDX",       # Industrials
    "MSFT", "INTC", "CSCO", "FLEX", "ACN",  # Information Technology
    "CRH", "NEM", "LIN", "IFF", "FCX",      # Materials
    "CBRE", "HST", "VTR", "DOC", "WY",      # Real Estate
    "PCG", "D", "XEL", "EIX", "SRE",        # Utilities
]

# --------------------------------------------------------------------------------------
# Date ranges (chronological train/test split to avoid look-ahead)
# --------------------------------------------------------------------------------------
START_DATE: str = "2005-01-01"
END_DATE: str = "2026-08-01"
SPLIT_DATE: str = "2020-01-01"  # train: dates < this; test: dates >= this

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

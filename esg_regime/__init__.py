"""esg_regime — a 3-state hidden-Markov regime detector for the ESG universe.

A companion to :mod:`esg_adaptive_rl`. It detects when the sustainable-investing
universe is calm, choppy, or stressed, so the RL allocator can later condition
its per-regime reward weights on the current market state.

Adapted from a production SPY regime model. The one substantive change is the
feature set: instead of SPX-option-implied VIX term structure, it uses a
self-contained *realised-volatility* term structure computed from the ESG
universe's own price history, so it needs no options data.

Submodules:
    - ``features``   : point-in-time, look-ahead-free feature engineering
    - ``classifier`` : the 3-state Gaussian-emission HMM (filtered posteriors,
                       hysteresis, deterministic stress guardrail)
    - ``regime``     : config + fit/predict detector matching the repo conventions
    - ``build_esg_index`` : builds a high-ESG price index from the real ESG scores
    - ``evaluate``   : train/test + walk-forward backtests and reporting
"""

from esg_regime.features import (  # noqa: F401
    ALL_FEATURES,
    CANDIDATE_FEATURES,
    CORE_FEATURES,
    LOCKED_FEATURES,
    compute_features,
)
from esg_regime.classifier import (  # noqa: F401
    REGIMES,
    RegimeClassifier,
    finalize_regimes,
    in_sample_regimes,
    walk_forward_regimes,
)
from esg_regime.regime import (  # noqa: F401
    MarketRegimeDetector,
    RegimeDetectorConfig,
    label_regimes,
    split_features_by_date,
)

__version__ = "0.1.0"

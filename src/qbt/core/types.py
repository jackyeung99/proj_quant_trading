from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import pandas as pd

SizeLike = Union[int, float, str, None]


@dataclass(frozen=True)
class WalkForwardSpec:
    train_size: int
    test_size: int
    step_size: int
    expanding: bool = True

@dataclass(frozen=True)
class RunSpec:
    # experiment identity
    strategy_name: str
    universe: str
    assets: list[str] = field(default_factory=list)
    tag: str | None = None


    # data + features
    data_path: str = None 
    features: dict[str, Any] = field(default_factory=dict)

    # strategy behavior
    params: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelBundle:
    model: Any
    feature_cols: list[str]
    ret_cols: list[str]
    trained_at: str
    train_end: str
    config_hash: str

    
@dataclass(frozen=True)
class BacktestSpec:
    # --- execution / timing ---
    weight_lag: int = 1
    rebalance: str = "D"
    train_freq: str = "D"
    transaction_cost_bps: float = 0.0

    # --- evaluation method (optional WF) ---
    use_walk_forward: bool = False
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    expanding: bool = True
    min_train: Optional[int] = 200

@dataclass(frozen=True)
class RunMeta:
    run_id: str
    strategy_name: str
    universe: str
    created_at_utc: str
    data_path: str
    weight_lag: int
    params: Dict[str, Any]
    tag: Optional[str] = None

@dataclass
class RunResult:
    meta: RunMeta
    timeseries: pd.DataFrame  # indexed by date
    metrics: Dict[str, Any]


@dataclass
class ModelInputs:
    ret: pd.DataFrame
    features: pd.DataFrame



# # ---------- Step specs ----------

# @dataclass(frozen=True)
# class IngestionSpec:
#     source: str                    # "alpaca", "polygon", "local_parquet", ...
#     raw_freq: str                  # "1Min", "5Min", etc
#     start: Optional[str] = None    # ISO date/time, optional
#     end: Optional[str] = None      # ISO date/time, optional
#     lookback_days: Optional[int] = None
#     extra: Dict[str, Any] = field(default_factory=dict)


# @dataclass(frozen=True)
# class ModelTableSpec:
#     # what you want to build
#     target_freq: str = "1D"        # daily targets even if raw is intraday
#     rv_freq: Optional[str] = None  # if you compute RV from intraday, ex "5Min"
#     cutoff_time: str = "16:00"     # cutoff defining "day" bucket
#     horizon: int = 1               # prediction horizon in bars at target_freq
#     features: Dict[str, Any] = field(default_factory=dict)
#     required_cols: list[str] = field(default_factory=list)

# @dataclass(frozen=True)
# class SignalSpec:
#     retrain_freq: str = "1D"
#     train_lookback_bars: int = 502
#     min_train_bars: int = 200
#     model_key_prefix: str = "models/"
#     meta_key_prefix: str = "models_meta/"
#     extra: Dict[str, Any] = field(default_factory=dict)


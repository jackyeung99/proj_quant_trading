from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd

@dataclass(frozen=True)
class RunSpec:
    strategy_name: str
    universe: str
    data_path: str                 # CSV or parquet path
    date_col: str = "date"
    ret_col: str = "ret"
    weight_lag: int = 1            # 1 = use weights decided at t-1 for return at t
    params: Dict[str, Any] = None  # strategy params (optional)
    tag: Optional[str] = None

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

@dataclass(frozen=True)
class WalkForwardSpec:
    train_size: int          # number of rows/days in train window
    test_size: int           # number of rows/days in test window
    step_size: int | None = None   # default = test_size
    expanding: bool = False        # if True, train expands instead of rolling
    min_train: int | None = None   # optional warmup guard
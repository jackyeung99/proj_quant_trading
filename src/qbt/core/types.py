from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import pandas as pd

SizeLike = Union[int, float, str, None]

@dataclass(frozen=True)
class RunSpec:
    strategy_name: str
    universe: str
    data_path: str                 # CSV or parquet path
    date_col: str = "date"
    assets: str = "ret"
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
    # Allow rows OR percent/fraction
    train_size: SizeLike          # int rows, float fraction, or "60%"
    test_size: SizeLike           # int rows, float fraction, or "20%"
    step_size: SizeLike = None    # default = test_size (after conversion)
    expanding: bool = False
    min_train: SizeLike = None    # optional warmup guard


@dataclass(frozen=True)
class ModelInputs:
    ret: pd.DataFrame        # [T x N] returns only (assets as cols)
    features: pd.DataFrame   # [T x ...] features (wide or multiindex)
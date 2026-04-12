from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import pandas as pd

SizeLike = Union[int, float, str, None]



@dataclass
class ModelBundle:
    model: Any
    ret_cols: list[str]
    asset_feature_cols: list[str]
    global_feature_cols: list[str]
    trained_at: str
    train_start: str
    train_end: str
    config_hash: str


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
    timeseries: pd.DataFrame
    metrics: Dict[str, Any]
    model_state: Dict[str, Any]  # learned/fitted outputs (tau*, weights, etc.)


@dataclass
class ModelInputs:
    ret: pd.DataFrame                         # [time × asset]
    asset_features: dict[str, pd.DataFrame]   # per-feature wide
    global_features: Optional[pd.DataFrame] = None   # global (optional)


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    side: str              # "long" | "short"
    market_value: float    # dollars
    current_price: float   # dollars
    avg_entry_price: float # dollars
    cost_basis: float      # dollars
    unrealized_pl: float   # dollars
    unrealized_plpc: float # fraction, e.g. 0.12



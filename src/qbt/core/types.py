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
    data: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)

    # strategy behavior
    params: dict[str, Any] = field(default_factory=dict)


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
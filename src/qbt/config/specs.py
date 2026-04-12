from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    stages: list[str]
    enabled: dict[str, bool]
    configs: dict[str, str]
    logging: dict[str, str | bool]
    storage: dict[str, str]

@dataclass(frozen=True)
class ConnectionSpec:
    name: str
    provider: str
    credential_source: str = "env"
    env: dict[str, str] = field(default_factory=dict)
    endpoints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSourceSpec:
    name: str
    connection: str
    kind: str
    symbols: list[str]
    interval: str
    params: dict[str, Any] = field(default_factory=dict)
    standardization: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    mode: str
    lookback_days: int
    table_name: str
    assets: list[str] = field(default_factory=list)
    sources: list[DatasetSourceSpec] = field(default_factory=list)
    aggregation: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategySpec:
    strategy_name: str
    strategy_class: str
    universe: str
    assets: list[str] = field(default_factory=list)
    tag: str | None = None

    input_table: str | None = None
    input_table_freq: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    training: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestSpec:
    weight_lag: int = 1
    rebalance: str = "D"
    train_freq: str = "D"
    transaction_cost_bps: float = 0.0

    use_walk_forward: bool = False
    train_size: int | None = None
    test_size: int | None = None
    test_start_years: int | None = None
    expanding: bool = True
    min_train: int | None = 200
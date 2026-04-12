from __future__ import annotations

from typing import Any

from qbt.config.specs import (
    BacktestSpec,
    ConnectionSpec,
    DatasetSourceSpec,
    DatasetSpec,
    PipelineSpec,
    StrategySpec,
)


def _require(
    mapping: dict[str, Any],
    key: str,
    ctx: str,
    *,
    allow_none: bool = False,
) -> Any:
    if not isinstance(mapping, dict):
        raise TypeError(
            f"Expected dict for {ctx}, got {type(mapping).__name__}"
        )

    if key not in mapping:
        available = ", ".join(mapping.keys())
        raise ValueError(
            f"Missing required key '{key}' in {ctx}. "
            f"Available keys: [{available}]"
        )

    value = mapping[key]

    if value is None and not allow_none:
        raise ValueError(f"Key '{key}' in {ctx} cannot be None")

    return value


def _as_dict(value: Any, ctx: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict in {ctx}, got {type(value).__name__}")
    return value


def _as_list(value: Any, ctx: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Expected list in {ctx}, got {type(value).__name__}")
    return value


def parse_pipeline_spec(raw: dict[str, Any]) -> PipelineSpec:
    name = str(_require(raw, "name", "pipeline config"))
    stages = [
        str(x)
        for x in _as_list(
            _require(raw, "stages", "pipeline config"),
            "pipeline.stages",
        )
    ]
    enabled_raw = _as_dict(
        _require(raw, "enabled", "pipeline config"),
        "pipeline.enabled",
    )
    configs_raw = _as_dict(
        _require(raw, "configs", "pipeline config"),
        "pipeline.configs",
    )
    logging_raw = _as_dict(raw.get("logging", {}), "pipeline.logging")
    storage_raw = _as_dict(
        _require(raw, "storage", "pipeline config"),
        "pipeline.storage",
    )

    return PipelineSpec(
        name=name,
        stages=stages,
        enabled={str(k): bool(v) for k, v in enabled_raw.items()},
        configs={str(k): str(v) for k, v in configs_raw.items()},
        logging={str(k): v for k, v in logging_raw.items()},
        storage={str(k): str(v) for k, v in storage_raw.items()},
    )


def parse_connection_specs(raw: dict[str, Any]) -> dict[str, ConnectionSpec]:
    root = _as_dict(
        _require(raw, "connections", "connections config"),
        "connections",
    )

    specs: dict[str, ConnectionSpec] = {}

    for name, cfg in root.items():
        cfg = _as_dict(cfg, f"connections.{name}")

        specs[name] = ConnectionSpec(
            name=name,
            provider=str(_require(cfg, "provider", f"connections.{name}")),
            credential_source=str(cfg.get("credential_source", "env")),
            env={
                str(k): str(v)
                for k, v in _as_dict(
                    cfg.get("env", {}),
                    f"connections.{name}.env",
                ).items()
            },
            endpoints={
                str(k): str(v)
                for k, v in _as_dict(
                    cfg.get("endpoints", {}),
                    f"connections.{name}.endpoints",
                ).items()
            },
        )

    return specs


def parse_dataset_source_spec(raw: dict[str, Any]) -> DatasetSourceSpec:
    raw = _as_dict(raw, "dataset source")

    source_name = str(raw.get("name", "?"))

    return DatasetSourceSpec(
        name=str(_require(raw, "name", "dataset source")),
        connection=str(_require(raw, "connection", f"dataset source '{source_name}'")),
        kind=str(_require(raw, "kind", f"dataset source '{source_name}'")),
        symbols=[
            str(x)
            for x in _as_list(
                _require(raw, "symbols", f"dataset source '{source_name}'"),
                f"dataset source '{source_name}'.symbols",
            )
        ],
        interval=str(_require(raw, "interval", f"dataset source '{source_name}'")),
        params=_as_dict(raw.get("params", {}), f"dataset source '{source_name}'.params"),
        standardization=_as_dict(
            raw.get("standardization", {}),
            f"dataset source '{source_name}'.standardization",
        ),
    )


def parse_dataset_spec(raw: dict[str, Any]) -> DatasetSpec:
    dataset = _as_dict(
        _require(raw, "dataset", "dataset config"),
        "dataset.dataset",
    )
    universe = _as_dict(raw.get("universe", {}), "dataset.universe")
    aggregation = _as_dict(raw.get("aggregation", {}), "dataset.aggregation")
    features = _as_dict(raw.get("features", {}), "dataset.features")
    output = _as_dict(
        _require(raw, "output", "dataset config"),
        "dataset.output",
    )

    sources_raw = _as_list(raw.get("sources", []), "dataset.sources")
    sources = [parse_dataset_source_spec(s) for s in sources_raw]

    return DatasetSpec(
        name=str(_require(dataset, "name", "dataset.dataset")),
        mode=str(dataset.get("mode", "append")),
        lookback_days=int(dataset.get("lookback_days", 1)),
        table_name=str(_require(output, "table_name", "dataset.output")),
        assets=[
            str(x)
            for x in _as_list(
                universe.get("assets", []),
                "dataset.universe.assets",
            )
        ],
        sources=sources,
        aggregation=aggregation,
        features=features,
        output=output,
    )


def parse_strategy_spec(raw: dict[str, Any]) -> StrategySpec:
    run = _as_dict(
        _require(raw, "run", "strategy config"),
        "strategy.run",
    )

    return StrategySpec(
        strategy_name=str(_require(run, "strategy_name", "strategy.run")),
        strategy_class=str(_require(run, "strategy_class", "strategy.run")),
        universe=str(_require(run, "universe", "strategy.run")),
        assets=[
            str(x)
            for x in _as_list(
                run.get("assets", []),
                "strategy.run.assets",
            )
        ],
        tag=run.get("tag"),
        input_table=run.get("input_table"),
        input_table_freq=run.get("input_table_freq", "1D"),
        params=_as_dict(raw.get("params", {}), "strategy.params"),
        training=_as_dict(raw.get("training", {}), "strategy.training"),
        execution=_as_dict(raw.get("execution", {}), "strategy.execution"),
        evaluation=_as_dict(raw.get("evaluation", {}), "strategy.evaluation"),
    )


def parse_backtest_spec(raw: dict[str, Any] | None) -> BacktestSpec | None:
    if raw is None:
        return None

    raw = _as_dict(raw, "backtest")

    return BacktestSpec(
        weight_lag=int(raw.get("weight_lag", 1)),
        rebalance=str(raw.get("rebalance", "D")),
        train_freq=str(raw.get("train_freq", "D")),
        transaction_cost_bps=float(raw.get("transaction_cost_bps", 0.0)),
        use_walk_forward=bool(raw.get("use_walk_forward", False)),
        train_size=None if raw.get("train_size") is None else int(raw["train_size"]),
        test_size=None if raw.get("test_size") is None else int(raw["test_size"]),
        test_start_years=None if raw.get("test_start_years") is None else int(raw["test_start_years"]),
        expanding=bool(raw.get("expanding", True)),
        min_train=None if raw.get("min_train") is None else int(raw["min_train"]),
    )
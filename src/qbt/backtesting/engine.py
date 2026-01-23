from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any
import pandas as pd
import numpy as np

from qbt.core.types import RunSpec, RunMeta, RunResult, WalkForwardSpec
from qbt.core.exceptions import DataError, InvalidRunSpec
from qbt.strategies.buy_hold import BuyHoldStrategy
from qbt.metrics.summary import compute_metrics
from qbt.backtesting.splitter import iter_walk_forward_splits
from qbt.execution.simulator import simulate_strategy_execution

_STRATEGY_MAP = {
    "BuyHold": BuyHoldStrategy,
}

def load_data(spec: RunSpec) -> pd.DataFrame:
    path = spec.data_path
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise InvalidRunSpec(f"Unsupported data_path: {path}")

    if spec.ret_col not in df.columns:
        raise DataError(f"Data must contain columns: {spec.date_col}, {spec.ret_col}")

    # df[spec.date_col] = pd.to_datetime(df[spec.date_col])
    # df = df.sort_values(spec.date_col).set_index(spec.date_col)
    df = df.sort_index(axis=0, ascending=True, inplace=False)
    df = df[[spec.ret_col]].rename(columns={spec.ret_col: "ret"})
    if df["ret"].isna().any():
        df = df.dropna()
    return df

def make_run_id(strategy: str, universe: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{strategy}_{universe}"

def run_backtest(spec: "RunSpec") -> "RunResult":
    wf_cfg = (spec.params or {}).get("walk_forward")  # or add spec.walk_forward
    if wf_cfg:
        wf = WalkForwardSpec(**wf_cfg)
        return run_backtest_walk_forward(spec, wf)

    # ---- existing single-shot behavior ----
    data = load_data(spec)

    if spec.strategy_name not in _STRATEGY_MAP:
        raise InvalidRunSpec(f"Unknown strategy_name={spec.strategy_name!r}")

    strat = _STRATEGY_MAP[spec.strategy_name]()
    w = strat.compute_weight(data, spec).astype(float)

    ts_df = simulate_strategy_execution(data, w, spec.weight_lag)

    run_id = make_run_id(spec.strategy_name, spec.universe)
    meta = RunMeta(
        run_id=run_id,
        strategy_name=spec.strategy_name,
        universe=spec.universe,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        data_path=spec.data_path,
        weight_lag=spec.weight_lag,
        params=spec.params or {},
        tag=spec.tag,
    )

    metrics = compute_metrics(ts_df["ret_net"])
    return RunResult(meta=meta, timeseries=ts_df, metrics=metrics)



def run_backtest_walk_forward(spec: "RunSpec", wf: WalkForwardSpec) -> "RunResult":
    data = load_data(spec)

    if spec.strategy_name not in _STRATEGY_MAP:
        raise InvalidRunSpec(f"Unknown strategy_name={spec.strategy_name!r}")

    strat = _STRATEGY_MAP[spec.strategy_name]()

    # this will hold weights for all dates (only populated on test windows)
    full_w = pd.Series(0.0, index=data.index, dtype=float)

    for train_idx, test_idx in iter_walk_forward_splits(data.index, wf):
        train = data.loc[train_idx]
        test = data.loc[test_idx]

        strat.fit(train, spec)

        # IMPORTANT: compute weights ONLY for the test period
        w_test = strat.compute_weight(test, spec).astype(float).reindex(test.index)

        # stitch into the full weight vector
        full_w.loc[test.index] = w_test

    ts_df = simulate_strategy_execution(data, full_w, spec.weight_lag)

    run_id = make_run_id(spec.strategy_name, spec.universe)
    meta = RunMeta(
        run_id=run_id,
        strategy_name=spec.strategy_name,
        universe=spec.universe,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        data_path=spec.data_path,
        weight_lag=spec.weight_lag,
        params={**(spec.params or {}), "walk_forward": True, "wf": wf.__dict__},
        tag=spec.tag,
    )

    metrics = compute_metrics(ts_df["ret_net"])
    return RunResult(meta=meta, timeseries=ts_df, metrics=metrics)


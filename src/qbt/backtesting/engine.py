from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any
import pandas as pd
import numpy as np

from qbt.core.types import RunSpec, RunMeta, RunResult
from qbt.core.exceptions import DataError, InvalidRunSpec
from qbt.strategies.buy_hold import BuyHoldStrategy
from qbt.metrics.summary import compute_metrics

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

    if spec.date_col not in df.columns or spec.ret_col not in df.columns:
        raise DataError(f"Data must contain columns: {spec.date_col}, {spec.ret_col}")

    df[spec.date_col] = pd.to_datetime(df[spec.date_col])
    df = df.sort_values(spec.date_col).set_index(spec.date_col)
    df = df[[spec.ret_col]].rename(columns={spec.ret_col: "ret"})
    if df["ret"].isna().any():
        df = df.dropna()
    return df

def make_run_id(strategy: str, universe: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{strategy}_{universe}"

def run_backtest(spec: RunSpec) -> RunResult:
    data = load_data(spec)

    if spec.strategy_name not in _STRATEGY_MAP:
        raise InvalidRunSpec(f"Unknown strategy_name={spec.strategy_name!r}")

    strat = _STRATEGY_MAP[spec.strategy_name]()
    w = strat.compute_weight(data, spec).astype(float)

    # central timing rule: apply lag so weight decided at t-1 applies to return at t
    w_lagged = w.shift(spec.weight_lag).fillna(0.0)

    ret_gross = w_lagged * data["ret"]
    ret_net = ret_gross.copy()  # MVP: no costs

    equity_gross = (1.0 + ret_gross).cumprod()
    equity_net = (1.0 + ret_net).cumprod()

    ts_df = pd.DataFrame(
        {
            "weight": w_lagged,
            "ret_gross": ret_gross,
            "ret_net": ret_net,
            "equity_gross": equity_gross,
            "equity_net": equity_net,
        },
        index=data.index,
    )

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

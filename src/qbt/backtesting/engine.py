from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field

import pandas as pd

from qbt.core.types import RunSpec, RunMeta, RunResult, BacktestSpec, ModelInputs
from qbt.metrics.summary import compute_metrics
from qbt.backtesting.splitter import iter_walk_forward_splits
from qbt.execution.simulator import simulate_strategy_execution
from qbt.data.dataloader import DataAdapter, DefaultDataAdapter

from qbt.strategies.strategy_registry import create_strategy, available_strategies
from qbt.strategies.strategy_base import Strategy

from qbt.core.logging import get_logger

logging = get_logger(__name__)


def make_run_id(strategy: str, universe: str) -> str:
    # ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + f"_{strategy}_{universe}"

    return run_id


def _normalize_weights_to_df(
    w: pd.Series | pd.DataFrame,
    index: pd.Index,
    assets: list[str],
) -> pd.DataFrame:
    """
    Normalize strategy output to a [T x N] DataFrame aligned to (index, assets).
    """
    if isinstance(w, pd.Series):
        # broadcast single series to all assets
        w_df = pd.concat({a: w for a in assets}, axis=1)
    else:
        w_df = w

    return w_df.reindex(index=index).reindex(columns=assets).fillna(0.0)


def build_weights(
    inputs: ModelInputs,
    strat: Strategy,
    spec: RunSpec,
    assets: list[str],
    bt: BacktestSpec
) -> pd.DataFrame:

 

    # ---- no walk-forward ----
    if bt is None:
        strat.fit(inputs, spec)
        w = strat.predict(inputs, spec)
        return _normalize_weights_to_df(w, index=inputs.ret.index, assets=assets)

    # ---- walk-forward ----
    full_w = pd.DataFrame(0.0, index=inputs.ret.index, columns=assets)

    for train_idx, test_idx in iter_walk_forward_splits(inputs.ret.index, bt):
        train_inputs = ModelInputs(
            ret=inputs.ret.loc[train_idx],
            features=inputs.features.loc[train_idx],
        )
        test_inputs = ModelInputs(
            ret=inputs.ret.loc[test_idx],
            features=inputs.features.loc[test_idx],
        )

        strat.fit(train_inputs, spec)
        w_test = strat.predict(test_inputs, spec)

        full_w.loc[test_idx, :] = _normalize_weights_to_df(
            w_test, index=test_inputs.ret.index, assets=assets
        )

    return full_w
@dataclass
class BacktestEngine:
    data_adapter: DataAdapter = field(default_factory=DefaultDataAdapter)

    def run(self, spec: RunSpec, bt: BacktestSpec) -> RunResult:
        strat = create_strategy(spec.strategy_name)

        raw = self.data_adapter.load(spec)
        required_features = strat.required_features(spec)

        inputs: ModelInputs = self.data_adapter.prepare(
            raw, spec, required_cols=required_features
        )

        assets = list(inputs.ret.columns)
   
        w = build_weights(
            inputs=inputs,
            strat=strat,
            spec=spec,
            assets=assets,
            bt=bt,
        )


        ts_df = simulate_strategy_execution(
            inputs.ret,
            w,
            weight_lag=bt.weight_lag,
            transaction_cost_bps=bt.transaction_cost_bps,
            # rebalance=bt.rebalance,
        )


        run_id = make_run_id(spec.strategy_name, spec.universe)
        meta = RunMeta(
            run_id=run_id,
            strategy_name=spec.strategy_name,
            universe=spec.universe,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            # store the YAML blocks so you can reproduce runs exactly
            data_path=str(spec.data.get("path") or spec.data.get("price_path") or ""),
            weight_lag=bt.weight_lag,
            params=spec.params or {},
            tag=spec.tag,
        )

        metrics = compute_metrics(ts_df["port_ret_gross"])
        print(metrics)
        return RunResult(meta=meta, timeseries=ts_df, metrics=metrics)
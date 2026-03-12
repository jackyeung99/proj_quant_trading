from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field

import pandas as pd
import numpy as np 

from qbt.core.types import RunSpec, RunMeta, RunResult, BacktestSpec, ModelInputs
from qbt.storage.storage import Storage
from qbt.metrics.summary import compute_portfolio_metrics
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
    bt: BacktestSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = inputs.ret.index

    full_w = pd.DataFrame(np.nan, index=idx, columns=assets)

    test_mask = pd.Series(False, index=idx)
    extra_blocks = []  # strategy-provided series for each test window

    for train_idx, test_idx in iter_walk_forward_splits(idx, bt):
        train_inputs = ModelInputs(ret=inputs.ret.loc[train_idx], features=inputs.features.loc[train_idx])
        test_inputs  = ModelInputs(ret=inputs.ret.loc[test_idx],  features=inputs.features.loc[test_idx])

        strat.fit(train_inputs, spec)

        # weights
        w_test = strat.predict(test_inputs, spec)
        full_w.loc[test_idx, :] = _normalize_weights_to_df(w_test, index=test_inputs.ret.index, assets=assets)

        # extra time series to persist (state_value, signal, tau_star, etc.)
        extra = strat.get_persisted_series(test_inputs=test_inputs, spec=spec)
        if extra is not None and not extra.empty:
            extra_blocks.append(extra.reindex(test_inputs.ret.index))

        test_mask.loc[test_idx] = True

    # concat once
    extra_df = pd.concat(extra_blocks).sort_index() if extra_blocks else pd.DataFrame(index=idx)

    return full_w, extra_df, test_mask

@dataclass
class BacktestEngine:
    storage: Storage
    data_adapter: DataAdapter = field(init=False)

    def __post_init__(self) -> None:
        self.data_adapter = DefaultDataAdapter(storage=self.storage)

    def run(self, spec: RunSpec, bt: BacktestSpec) -> RunResult:
        strat = create_strategy(spec.strategy_name)

        raw = self.data_adapter.load(spec)
        required_features = strat.required_features(spec)
        inputs: ModelInputs = self.data_adapter.prepare(raw, spec, required_cols=required_features)

        assets = list(inputs.ret.columns)

        w_full, state_full, test_mask = build_weights(
            inputs=inputs, strat=strat, spec=spec, assets=assets, bt=bt
        )

        # ---- TEST-ONLY slice ----
        ret_test = inputs.ret.loc[test_mask]
        w_test = w_full.loc[test_mask].fillna(0.0)         # safe: only test rows remain
        state_test = state_full.loc[state_full.index.intersection(ret_test.index)]

        ts_df = simulate_strategy_execution(
            ret_test,
            w_test,
            weight_lag=bt.weight_lag,
            transaction_cost_bps=bt.transaction_cost_bps,
            asset_return_type="log",
            # add_buy_and_hold=True, buy_and_hold_asset=assets[0]  # if you added it
        )

        # append model state onto test timeseries (no join surprises)
        ts_df = ts_df.join(state_test, how="left")

        run_id = make_run_id(spec.strategy_name, spec.universe)
        meta = RunMeta(
            run_id=run_id,
            strategy_name=spec.strategy_name,
            universe=spec.universe,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            data_path=str(spec.data_path) or "",
            weight_lag=bt.weight_lag,
            params=spec.params or {},
            tag=spec.tag,
        )

        metrics = compute_portfolio_metrics(
            ts_df,
            ann_factor=252,
            return_type="simple",  # or "log" if needed
            col_signal="signal",   # your 0/1 column
        )

        return RunResult(meta=meta, timeseries=ts_df, metrics=metrics, model_state=state_test)
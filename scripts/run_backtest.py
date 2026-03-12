import yaml
from pathlib import Path
from typing import Optional, Iterable, Dict, Any
from itertools import product
import copy

from qbt.core.types import RunSpec, BacktestSpec
from qbt.backtesting.engine import BacktestEngine
from qbt.data.dataloader import DefaultDataAdapter
from qbt.storage.storage import LocalStorage
from qbt.storage.paths import StoragePaths
from qbt.storage.artifacts import BacktestStore

from qbt.core.logging import setup_logging


def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def deep_update(d: dict, u: dict) -> dict:
    """
    Recursively update dict d with dict u (returns a new dict).
    """
    out = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def run_backtest(
    *,
    spec: RunSpec,
    bt_spec: BacktestSpec,
    engine: BacktestEngine,
    artifact_store: BacktestStore,
) -> None:
    result = engine.run(spec, bt_spec)

    artifact_store.write_run(result.meta, result.timeseries, result.metrics)
    print(
        "Wrote run:",
        result.meta.run_id,
        "|",
        spec.strategy_name,
        "|",
        spec.universe,
        "| tag:",
        spec.tag,
        "| Sharpe:", 
        result.metrics['gross_sharpe']
    )


def main():
    # --- storage ---
    storage = LocalStorage(base_dir=Path("."))
    paths = StoragePaths()
    experiment_name = 'macro_variables'
    artifact_store = BacktestStore(storage, paths, experiment=experiment_name)

    # --- engine ---
    bt = BacktestEngine(storage=storage)

    # --- backtesting settings ---
    bt_cfg = load_yaml("configs/bt_method.yaml")
    bt_spec = BacktestSpec(**bt_cfg)

    base_state = load_yaml("configs/strategies/run_state.yaml")['run']
    base_state["data_path"] = "data/gold/freq=1D/tag=experiment/table.parquet"

    state_vars = ["XLE_rvol", "DCOILWTICO", "DHHNGSP", "GASREGCOVW", "OVXCLS"]
    weight_types = ["binary", "mean_var"]
    gamma_types = [1, 5, 10]
    min_frac = [0, 0.05, .1, .15]

    for state_var, weight_type, gamma, outlier_cap in product(state_vars, weight_types, gamma_types, min_frac):
            
        cfg = deep_update(
            base_state,
            {
                "params": { 
                    "min_frac": outlier_cap,
                    "state_var": state_var,
                    "weight_allocation": weight_type,
                    "gamma": gamma
                }   
            },
        )

        
        spec = RunSpec(**cfg)

        run_backtest(
            spec=spec,
            bt_spec=bt_spec,
            engine=bt,
            artifact_store=artifact_store,
        )

    



    


if __name__ == "__main__":
    setup_logging()
    main()

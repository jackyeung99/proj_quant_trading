import yaml
from pathlib import Path
from typing import Optional, Iterable, Dict, Any
import copy

from qbt.core.types import RunSpec, BacktestSpec
from qbt.backtesting.engine import BacktestEngine
from qbt.storage.storage import LocalStorage
from qbt.storage.paths import StoragePaths
from qbt.storage.artifacts import ArtifactsStore
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


def run_spec(cfg: dict, bt_spec_df: dict, store: ArtifactsStore, bt: BacktestEngine) -> None:
    spec = RunSpec(**cfg)
    bt_spec = BacktestSpec(**bt_spec_df)


    result = bt.run(spec, bt_spec)
    store.write_run(result.meta, result.timeseries, result.metrics)
    print("Wrote run:", result.meta.run_id, "|", spec.strategy_name, "|", spec.universe, "| tag:", spec.tag)


def main():
    # --- storage ---
    storage = LocalStorage(base_dir=Path("."))
    paths = StoragePaths()
    store = ArtifactsStore(storage, paths)

    # --- engine ---
    bt = BacktestEngine()

    # --- backtesting settings ---
    bt_cfg = load_yaml( "configs/bt_method.yaml")


    # --- base configs (template YAMLs) ---
    base_buyhold = load_yaml("configs/strategies/run_buyhold.yaml")
    run_spec(base_buyhold, bt_cfg, store, bt)


    base_state = load_yaml("configs/strategies/run_state.yaml")
    state_vars = [
                "XLE_rvol", "XLE_garch_sigma_ann_next", "XLE_ewma_sigma_ann_next"
                # "XLE_jump_var", "XLE_ewma_vol_20", "XLE_mom_20", "XLE_mom_60", "XLE_trend_slope_60", "XLE_pk_vol"
                  ]
    
    cutoffs = ["data/gold/freq=1D/tag=trade-same-day-12/table.parquet", "data/gold/freq=1D/tag=trade-same-day-3/table.parquet", "data/gold/freq=1D/tag=trade-next-day/table.parquet"]
    times = ["12 PM ET", "3PM ET", "9 AM ET Next Day"]

    for cutoff_path, time  in zip(cutoffs, times):
    
        for sv in state_vars:
            cfg = copy.deepcopy(base_state)

            cfg["data_path"] = cutoff_path
            cfg["params"]["state_var"] = sv

            # NEW PARAM: you choose the param name your strategy/backtest reads
            cfg["params"]["trade_time_et"] = time  # e.g., 3.0, 12.0, or None

            # tag includes both cutoff + state var
            cfg["tag"] = f"{cfg.get('tag','run')}_{time}_{sv}"

            run_spec(cfg, bt_cfg, store, bt)



    


if __name__ == "__main__":
    setup_logging()
    main()

import yaml
from pathlib import Path
from typing import Optional, Iterable, Dict, Any
import copy

from qbt.core.types import RunSpec, WalkForwardSpec
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


def run_spec(cfg: dict, wf: Optional[WalkForwardSpec], store: ArtifactsStore, bt: BacktestEngine) -> None:
    spec = RunSpec(**cfg)
    result = bt.run(spec, wf)
    store.write_run(result.meta, result.timeseries, result.metrics)
    print("Wrote run:", result.meta.run_id, "|", spec.strategy_name, "|", spec.universe, "| tag:", spec.tag)

def main():
    # --- storage ---
    storage = LocalStorage(base_dir=Path("."))
    paths = StoragePaths(root="results")
    store = ArtifactsStore(storage, paths)

    # --- engine ---
    bt = BacktestEngine()

    # --- walk-forward ---
    wf_path = "configs/bt_method.yaml"
    wf_cfg = load_yaml(wf_path)
    wf = WalkForwardSpec(**wf_cfg)

    # --- base configs (template YAMLs) ---
    base_buyhold = load_yaml("configs/strategies/run_buyhold.yaml")
    base_state = load_yaml("configs/strategies/run_state.yaml")

    # --- common overrides (XLE only) ---
    common_xle = {
        "universe": "XLE",
        "assets": ["ret_xle"],
        "tag": "sweep_xle",
    }

    # --- BuyHold (XLE only) ---
    cfg = deep_update(base_buyhold, common_xle)
    cfg.setdefault("tag", "sweep_xle_buyhold")
    run_spec(cfg, wf, store, bt)

    # --- StateSignal (XLE only) ---
    xle_state_vars = ["rv_idio", "rv_spy", "rv_xle"]
    for state_var in xle_state_vars:
        v = {
            "universe": "XLE",
            "assets": ["ret_xle"],
            "params": {"gamma": 5, "lag_state": 1, "state_var": state_var},
            "tag": f"xle_state_{state_var}",
        }
        cfg = deep_update(base_state, v)
        run_spec(cfg, wf, store, bt)


if __name__ == "__main__":
    setup_logging()
    main()

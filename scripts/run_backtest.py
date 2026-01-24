import yaml
from pathlib import Path
from typing import Optional, Iterable, Dict, Any
import copy

from qbt.core.types import RunSpec, WalkForwardSpec
from qbt.backtesting.engine import BacktestEngine
from qbt.storage.storage import LocalStorage
from qbt.storage.paths import StoragePaths
from qbt.storage.artifacts import ArtifactsStore


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

    # --- common overrides for SPY ---
    common_spy = {
        "universe": "SPY",
        "assets": ["ret_spy"],
        "tag": "sweep_spy",
    }

    # --- variant grids ---
    # BuyHold variants (example: equal weight vs explicit weight vs normalization)
    buyhold_variants = [
        {"universe": 'SPY'}
    ]

    # StateSignal variants (example: sweep gamma and lag_state)
    state_variants = []
    for gamma in [2.0, 5.0, 10.0]:
        for lag_state in [1, 2, 5]:
            state_variants.append({
                "params": {
                    "gamma": gamma,
                    "lag_state": lag_state,
                    # if your feature name differs for SPY, change here:
                    # e.g., rv_spy instead of rv_xle
                    "state_var": "rv_spy",
                },
                "tag": f"sweep_spy_state_gamma={gamma}_lag={lag_state}",
            })

    # --- run them ---
    # BuyHold sweep
    for v in buyhold_variants:
        cfg = deep_update(base_buyhold, common_spy)
        cfg = deep_update(cfg, v)
        # give a unique tag if not provided
        cfg.setdefault("tag", "sweep_spy_buyhold")
        run_spec(cfg, wf, store, bt)

    # State sweep
    for v in state_variants:
        cfg = deep_update(base_state, common_spy)
        cfg = deep_update(cfg, v)
        run_spec(cfg, wf, store, bt)


if __name__ == "__main__":
    main()

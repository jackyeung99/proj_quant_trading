import yaml
from pathlib import Path

from qbt.core.types import RunSpec, WalkForwardSpec
from qbt.backtesting.engine import run_backtest
from qbt.storage.storage import LocalStorage
from qbt.storage.paths import StoragePaths
from qbt.storage.artifacts import ArtifactsStore


def run_one(strat_cfg_path: str, walkforward_cfg_path: str):
    cfg = yaml.safe_load(Path(strat_cfg_path).read_text())
    spec = RunSpec(**cfg)

    wf_cfg = yaml.safe_load(Path(walkforward_cfg_path).read_text())
    wf = WalkForwardSpec(**wf_cfg)

    result = run_backtest(spec, wf)

    storage = LocalStorage(base_dir=Path("."))
    paths = StoragePaths(root="results")
    store = ArtifactsStore(storage, paths)

    store.write_run(result.meta, result.timeseries, result.metrics)
    print("Wrote run:", result.meta.run_id, "strategy:", spec.strategy_name)


def main():
    wf_path = "configs/bt_method.yaml"
    run_one("configs/strategies/run_buyhold.yaml", wf_path)
    run_one("configs/strategies/run_state.yaml", wf_path)


if __name__ == "__main__":
    main()

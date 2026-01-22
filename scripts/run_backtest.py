import yaml
from pathlib import Path

from qbt.core.types import RunSpec
from qbt.backtesting.engine import run_backtest
from qbt.storage.storage import LocalStorage
from qbt.storage.paths import StoragePaths
from qbt.storage.artifacts import ArtifactsStore

def main():
    cfg = yaml.safe_load(Path("configs/run_buyhold.yaml").read_text())
    spec = RunSpec(**cfg)


    result = run_backtest(spec)

    storage = LocalStorage(base_dir=Path("."))  # project root
    paths = StoragePaths(root="results")
    store = ArtifactsStore(storage, paths)



    store.write_run(result.meta, result.timeseries, result.metrics)
    print("Wrote run:", result.meta.run_id)

if __name__ == "__main__":
    main()

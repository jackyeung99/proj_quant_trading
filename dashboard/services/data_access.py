from pathlib import Path
import pandas as pd

from qbt.storage.storage import LocalStorage
from qbt.storage.paths import StoragePaths
from qbt.storage.artifacts import ArtifactsStore


class StoreContext:
    def __init__(self, store: ArtifactsStore):
        self.store = store


def build_store(base_dir: Path, results_root: str) -> StoreContext:
    storage = LocalStorage(base_dir=base_dir)
    paths = StoragePaths(root=results_root)
    store = ArtifactsStore(storage, paths)
    return StoreContext(store)


def load_runs(ctx: StoreContext) -> pd.DataFrame:
    runs = ctx.store.read_runs()
    if runs.empty:
        return runs
    runs = runs.copy()
    runs["label"] = runs["run_id"] + " | " + runs["strategy_name"] + " | " + runs["universe"]
    return runs.sort_values("created_at_utc", ascending=False)


def safe_read_metrics(ctx: StoreContext) -> pd.DataFrame:
    if hasattr(ctx.store, "read_metrics"):
        try:
            return ctx.store.read_metrics()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def read_timeseries_for_run(ctx: StoreContext, run_id: str):
    runs = ctx.store.read_runs()
    row = runs.loc[runs["run_id"] == run_id].iloc[0]
    ts = ctx.store.read_timeseries(row["strategy_name"], row["universe"], run_id)
    return row, ts

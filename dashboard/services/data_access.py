from pathlib import Path
import pandas as pd
import json

from qbt.storage.storage import LocalStorage
from qbt.storage.paths import StoragePaths
from qbt.storage.artifacts import ArtifactsStore


class StoreContext:
    def __init__(self, store: ArtifactsStore):
        self.store = store

def _flatten(d, prefix=""):
    """Flatten nested dict into dot keys: {'a': {'b': 1}} -> {'a.b': 1}"""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out

def read_meta_for_run(ctx: StoreContext, run_id: str) -> dict:
    # Assuming LocalStorage paths like: results/runs/<run_id>/meta.json
    # If your StoragePaths exposes a method, use that instead. 
    return  ctx.store.read_meta(run_id)
    

def meta_params_table(meta: dict) -> pd.DataFrame:
    if not isinstance(meta, dict) or not meta:
        return pd.DataFrame([{"parameter": "info", "value": "meta is empty"}])

    # Try known param containers first
    for key in ["params", "strategy_params", "model_params", "spec"]:
        if isinstance(meta.get(key), dict) and meta[key]:
            candidate = meta[key]
            break
    else:
        # fallback: use entire meta minus obvious non-params
        candidate = {k: v for k, v in meta.items() if k not in ["run_id", "created_at", "created_at_utc"]}

    flat = _flatten(candidate)

    if not flat:
        return pd.DataFrame([{"parameter": "info", "value": "No parameters found in meta.json"}])

    df = pd.DataFrame(
        [{"parameter": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v}
         for k, v in sorted(flat.items())]
    )
    return df


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

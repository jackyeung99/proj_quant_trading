from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any, Optional, List
import pandas as pd
import json

from qbt.core.exceptions import StorageError
from qbt.core.types import RunMeta
from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths

_REQUIRED_TS_COLS = ["port_ret_gross", "port_ret_net", "equity_gross", "equity_net"]

class ArtifactsStore:
    def __init__(self, storage: Storage, paths: StoragePaths):
        self.storage = storage
        self.paths = paths

    def _validate_timeseries(self, ts: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED_TS_COLS if c not in ts.columns]
        if missing:
            raise StorageError(f"Timeseries missing columns: {missing}")
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise StorageError("Timeseries index must be a DatetimeIndex.")
        if not ts.index.is_monotonic_increasing:
            raise StorageError("Timeseries index must be increasing.")

    def write_run(self, meta: RunMeta, timeseries: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        self._validate_timeseries(timeseries)

        # 1) write timeseries
        ts_key = self.paths.run_timeseries_key(meta.strategy_name, meta.universe, meta.run_id)
        ts_to_write = timeseries.copy()
        ts_to_write.insert(0, "run_id", meta.run_id)
        self.storage.write_parquet(ts_to_write, ts_key)

        # 2) write meta json
        self.storage.write_json(asdict(meta), self.paths.run_meta_key(meta.run_id))

        # 3) upsert registry row (runs.parquet)
        runs_key = self.paths.runs_key()
        row = {
            "run_id": meta.run_id,
            "strategy_name": meta.strategy_name,
            "universe": meta.universe,
            "created_at_utc": meta.created_at_utc,
            "data_path": meta.data_path,
            "weight_lag": meta.weight_lag,
            "tag": meta.tag,
            "params": json.dumps(meta.params or {}, sort_keys=True),
        }
        runs_df = self.storage.read_parquet(runs_key) if self.storage.exists(runs_key) else pd.DataFrame()
        runs_df = self._upsert(runs_df, row, key="run_id")
        self.storage.write_parquet(runs_df, runs_key)

        # 4) upsert metrics row (metrics.parquet)
        metrics_key = self.paths.metrics_key()
        mrow = {"run_id": meta.run_id, **metrics}
        metrics_df = self.storage.read_parquet(metrics_key) if self.storage.exists(metrics_key) else pd.DataFrame()
        metrics_df = self._upsert(metrics_df, mrow, key="run_id")
        self.storage.write_parquet(metrics_df, metrics_key)

    def read_runs(self) -> pd.DataFrame:
        key = self.paths.runs_key()
        return self.storage.read_parquet(key) if self.storage.exists(key) else pd.DataFrame()

    def read_meta(self, run_id) -> pd.DataFrame:
        key = self.paths.run_meta_key(run_id)
        return self.storage.read_json(key) if self.storage.exists(key) else {}

    def read_metrics(self) -> pd.DataFrame:
        key = self.paths.metrics_key()
        return self.storage.read_parquet(key) if self.storage.exists(key) else pd.DataFrame()

    def read_timeseries(self, strategy: str, universe: str, run_id: str) -> pd.DataFrame:
        key = self.paths.run_timeseries_key(strategy, universe, run_id)
        df = self.storage.read_parquet(key)
        # return date-indexed
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df

    @staticmethod
    def _upsert(df: pd.DataFrame, row: Dict[str, Any], key: str) -> pd.DataFrame:
        row_df = pd.DataFrame([row])
        if df.empty:
            return row_df
        if key in df.columns:
            df = df[df[key] != row[key]]
        return pd.concat([df, row_df], ignore_index=True)

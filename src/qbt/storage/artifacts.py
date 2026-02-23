from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any, Optional, List
import pandas as pd
import json
import re


from qbt.core.exceptions import StorageError
from qbt.core.types import ModelBundle
from qbt.core.types import RunMeta
from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths

_REQUIRED_TS_COLS = ["port_ret_gross", "port_ret_net", "equity_gross", "equity_net"]

class BacktestStore:
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

class LiveStore:
    """
    Live artifact store (signal -> execution).

    Mental model:
      - Signal writes weights: latest pointer + immutable snapshot per asof
      - Execution uses guards: lock + last_exec (idempotency)
      - Execution writes: orders batch (planned) + trades batch (ledger)
    """

    def __init__(self, storage: Storage, paths: StoragePaths):
        self.storage = storage
        self.paths = paths

    # ---------------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------------
    def read_model(self, strategy: str, universe: str) -> Optional[ModelBundle]:
        key = self.paths.model_latest_pkl_key(strategy=strategy, universe=universe)
        return self.storage.read_pickle(key) if self.storage.exists(key) else None

    def read_model_meta(self, strategy: str, universe: str) -> Dict[str, Any]:
        key = self.paths.model_latest_meta_key(strategy=strategy, universe=universe)
        return self.storage.read_json(key) if self.storage.exists(key) else {}
    

    def read_all_model_meta(self, strategy: str, universe: str) -> pd.DataFrame:
        """
        Read all model meta snapshots for (strategy, universe) and return a time-series DF.

        Expected snapshot layout (recommended):
        models/{strategy}/{universe}/snapshots/{snapshot_id}/meta.json

        Returns:
        DataFrame indexed by session_date (tz-naive midnight),
        with columns including:
            - snapshot_id
            - trained_at_utc
            - config_hash
            - train_* fields
            - flattened strategy_meta fields (tau_star, w_low, ...)
        """
        prefix = self.paths.model_snapshot_prefix(strategy, universe)
        keys = self.storage.list(prefix)

        if not keys:
            return pd.DataFrame()

        # keep only meta.json files
        meta_keys = [k for k in keys if str(k).endswith("meta.json")]
        if not meta_keys:
            return pd.DataFrame()

        def _parse_snapshot_id_from_key(k: str) -> str | None:
            # works for .../snapshots/<snapshot_id>/meta.json or .../snapshot_id=<...>/meta.json
            s = str(k)
            m = re.search(r"/snapshots/([^/]+)/meta\.json$", s)
            if m:
                return m.group(1)
            m = re.search(r"/snapshot_id=([^/]+)/meta\.json$", s)
            if m:
                return m.group(1)
            return None

        rows: list[dict] = []

        for k in meta_keys:
            try:
                meta = self.storage.read_json(k)
            except Exception:
                continue

            if not meta:
                continue

            snapshot_id = meta.get("snapshot_id") or _parse_snapshot_id_from_key(str(k))

            trained_at_utc = meta.get("trained_at_utc") or meta.get("trained_at")
            trained_at_utc = None if trained_at_utc in (None, "", "unknown") else str(trained_at_utc)

            market_tz = meta.get("market_tz") or "America/New_York"

            trained_at_utc = meta.get("trained_at_utc")

            # stored as UTC stamps (good to keep tz-aware)
            train_start_asof_utc = meta.get("train_start_asof_utc")
            train_end_asof_utc = meta.get("train_end_asof_utc")

            # stored as session_date-ish strings but currently include tz "+00:00"
            train_start_session = meta.get("train_start_session_date")
            train_end_session = meta.get("train_end_session_date")


            # strategy_meta may be nested
            strat_meta = meta.get("strategy_meta", {}) or {}
            if not isinstance(strat_meta, dict):
                strat_meta = {}

            row = {
                # join/index fields
                "session_date": train_end_session,

                # identity/lineage
                "snapshot_id": snapshot_id,
                "config_hash": meta.get("config_hash"),

                # timing
                "trained_at_utc": trained_at_utc,
                "train_start_asof_utc": train_start_asof_utc,
                "train_end_asof_utc": train_end_asof_utc,
                "train_start_session_date": train_start_session,
                "train_end_session_date": train_end_session,

                # config
                "retrain_freq": meta.get("retrain_freq"),
                "train_lookback_bars": meta.get("train_lookback_bars"),
                "min_train_bars": meta.get("min_train_bars"),
                "bundle_version": meta.get("bundle_version"),
                "market_tz": market_tz,
                "cutoff_hour": meta.get("cutoff_hour"),

                # flattened strategy meta
                **strat_meta,
            }
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Build session_date index (tz-naive midnight)
        # IMPORTANT: don't do utc=True here; this is a daily label, not an event timestamp.
        df["session_date"] = pd.to_datetime(df["train_end_session_date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["session_date"]).set_index("session_date").sort_index()

        # De-dup: if multiple snapshots map to same session_date, keep the last by trained_at_utc
        if "trained_at_utc" in df.columns:
            ta = pd.to_datetime(df["trained_at_utc"], utc=True, errors="coerce")
            df = df.assign(_trained_at_sort=ta).sort_values(["session_date", "_trained_at_sort"])
            df = df.drop(columns=["_trained_at_sort"])

        df = df[~df.index.duplicated(keep="last")]

        return df

    def write_model(
        self,
        *,
        strategy: str,
        universe: str,
        bundle: ModelBundle,
        meta: Dict[str, Any],
        params: Dict[str, Any] | None = None,
        snapshot: bool = True,
    ) -> None:

        # -------------------------
        # LATEST (fast live access)
        # -------------------------
        self.storage.write_pickle(
            bundle,
            self.paths.model_latest_pkl_key(strategy=strategy, universe=universe ),
        )

        self.storage.write_json(
            meta,
            self.paths.model_latest_meta_key(strategy=strategy, universe=universe),
        )

        # -------------------------
        # SNAPSHOT (versioned)
        # -------------------------
        if snapshot:
            trained_at_utc  = str(meta.get("trained_at_utc", "unknown"))

            # model snapshot
            self.storage.write_pickle(
                bundle,
                self.paths.model_snapshot_pkl_key(strategy=strategy, universe=universe, trained_at=trained_at_utc),
            )

            # meta snapshot
            self.storage.write_json(
                meta,
                self.paths.model_snapshot_meta_key(strategy=strategy, universe=universe, trained_at=trained_at_utc),
            )

    # ---------------------------------------------------------------------
    # Weights (signal output): latest + snapshot(asof)
    # ---------------------------------------------------------------------
    def write_weights(
        self,
        *,
        strategy: str,
        universe: str,
        latest_w: pd.DataFrame,         # single-row df indexed by asof
        snapshot: bool = True,
    ) -> None:
        """
        Store weights as:
          - latest.parquet (overwrite)
          - snapshots/asof=...parquet (optional, overwrite-by-key)
        """
        if latest_w.shape[0] != 1:
            raise StorageError("latest_w must be a single-row DataFrame indexed by asof.")

        row = latest_w.copy()
        row.index.name = row.index.name or "asof"
        asof = str(row.index[0])

        # write latest pointer (overwrite)
        latest_key = self.paths.weights_latest_key(strategy, universe)
        self.storage.write_parquet(row, latest_key)

        # write immutable snapshot (idempotent overwrite on same asof)
        if snapshot:
            snap_key = self.paths.weights_snapshot_key(strategy, universe, asof=asof)
            self.storage.write_parquet(row, snap_key)

    def read_weights(self, strategy: str, universe: str) -> pd.DataFrame:
        """
        Read latest weights (single-row df indexed by asof).
        """
        key = self.paths.weights_latest_key(strategy, universe)
        return self.storage.read_parquet(key) if self.storage.exists(key) else pd.DataFrame()

    def read_weights_snapshot(self, strategy: str, universe: str, *, asof: str) -> pd.DataFrame:
        """
        Read a specific asof snapshot (single-row df).
        """
        key = self.paths.weights_snapshot_key(strategy, universe, asof=asof)
        return self.storage.read_parquet(key) if self.storage.exists(key) else pd.DataFrame()
    

    def read_all_weights(self, strategy: str, universe: str) -> pd.DataFrame:
        """
        Read all weight snapshots for (strategy, universe) and return a
        time-series DataFrame sorted by asof.

        Returns:
            DataFrame indexed by asof (datetime-like), columns = tickers
        """

        prefix = self.paths.weights_snapshots_prefix(strategy, universe)

        # list snapshot files (implementation depends on your storage backend)
        keys = self.storage.list(prefix)

        if not keys:
            return pd.DataFrame()

        parts = []

        for k in keys:
            try:
                df = self.storage.read_parquet(k)
                if not df.empty:
                    parts.append(df)
            except Exception:
                continue

        if not parts:
            return pd.DataFrame()

        out = pd.concat(parts)

        # ensure index name
        out.index.name = out.index.name or "asof"

        # convert to datetime if possible
        try:
            out.index = pd.to_datetime(out.index, utc=True)
        except Exception:
            pass

        out = (
            out
            .sort_index()
            .drop_duplicates(keep="last")
        )

        return out

    # ---------------------------------------------------------------------
    # Execution guards: lock + last_exec
    # ---------------------------------------------------------------------
    def read_lock(self, *, strategy: str, universe: str) -> Dict[str, Any]:
        key = self.paths.exec_lock_key(strategy, universe)
        return self.storage.read_json(key) if self.storage.exists(key) else {}

    def write_lock(self, *, strategy: str, universe: str, meta: Dict[str, Any]) -> None:
        key = self.paths.exec_lock_key(strategy, universe)
        self.storage.write_json(meta, key)

    def clear_lock(self, *, strategy: str, universe: str) -> None:
        """
        Best-effort. If Storage doesn't support delete yet, keep a tombstone.
        """
        key = self.paths.exec_lock_key(strategy, universe)
        if hasattr(self.storage, "delete") and callable(getattr(self.storage, "delete")):
            self.storage.delete(key)  # type: ignore[attr-defined]
        else:
            self.storage.write_json({"locked": False}, key)

    def read_last_exec(self, *, strategy: str, universe: str) -> Dict[str, Any]:
        key = self.paths.last_exec_key(strategy, universe)
        return self.storage.read_json(key) if self.storage.exists(key) else {}

    def write_last_exec(self, *, strategy: str, universe: str, meta: Dict[str, Any]) -> None:
        key = self.paths.last_exec_key(strategy, universe)
        self.storage.write_json(meta, key)

    def already_executed_asof(self, *, strategy: str, universe: str, asof: str) -> bool:
        m = self.read_last_exec(strategy=strategy, universe=universe)
        return (m.get("asof") == asof) and (m.get("status") in {"submitted", "completed"})

    # ---------------------------------------------------------------------
    # Planned orders (optional but useful for idempotency/auditing)
    # ---------------------------------------------------------------------
    def write_orders_batch(
        self,
        *,
        strategy: str,
        universe: str,
        orders: pd.DataFrame,
        batch_id: str,
        timestamp_col: str = "timestamp",
    ) -> str:
        if orders.empty:
            raise StorageError("orders is empty")
        if timestamp_col not in orders.columns:
            raise StorageError(f"orders must include '{timestamp_col}' column")

        ts = pd.to_datetime(orders[timestamp_col].iloc[0], utc=True, errors="coerce")
        if pd.isna(ts):
            raise StorageError(f"Invalid {timestamp_col} value: {orders[timestamp_col].iloc[0]}")

        date = ts.strftime("%Y-%m-%d")
        key = self.paths.orders_batch_key(strategy, universe, date=date, batch_id=batch_id)
        self.storage.write_parquet(orders, key)
        return key

    def read_orders(
        self,
        *,
        strategy: str,
        universe: str,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        root = self.paths.orders_root(strategy, universe)
        key = f"{root}/date={date}" if date else root
        return self.storage.read_parquet(key) if self.storage.exists(key) else pd.DataFrame()

    # ---------------------------------------------------------------------
    # Trades ledger (executed): batches dataset
    # ---------------------------------------------------------------------
    def write_trades_batch(
        self,
        *,
        strategy: str,
        universe: str,
        trades: pd.DataFrame,
        batch_id: str,
        timestamp_col: str = "timestamp",
    ) -> str:
        if trades.empty:
            raise StorageError("trades is empty")
        if timestamp_col not in trades.columns:
            raise StorageError(f"trades must include '{timestamp_col}' column")

        ts = pd.to_datetime(trades[timestamp_col].iloc[0], utc=True, errors="coerce")
        if pd.isna(ts):
            raise StorageError(f"Invalid {timestamp_col} value: {trades[timestamp_col].iloc[0]}")

        date = ts.strftime("%Y-%m-%d")
        key = self.paths.trades_batch_key(strategy, universe, date=date, batch_id=batch_id)
        self.storage.write_parquet(trades, key)
        return key

    def read_trades(
        self,
        *,
        strategy: str,
        universe: str,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        root = self.paths.trades_root(strategy, universe)
        key = f"{root}/date={date}" if date else root
        return self.storage.read_parquet(key) if self.storage.exists(key) else pd.DataFrame()
    

    def write_portfolio_performance(
        self,
        *,
        strategy: str,
        universe: str,
        df: pd.DataFrame,
        metrics: dict,
    ) -> None:
        if df is None or df.empty:
            return

        now_utc = pd.Timestamp.now(tz="UTC")

        now_iso = now_utc.isoformat()

        # -----------------------------
        # 1) Time series parquet (latest)
        # -----------------------------
        ts_key = self.paths.performance_ts(strategy=strategy, universe=universe)

        x = df.sort_index()
        self.storage.write_parquet(x, ts_key)

        # -----------------------------
        # 2) Metrics json (latest)
        # -----------------------------
        metrics_key = self.paths.performance_metrics(strategy=strategy, universe=universe)

        metrics_out = dict(metrics or {})
        metrics_out["generated_at_utc"] = now_iso
        metrics_out["strategy"] = strategy
        metrics_out["universe"] = universe
        metrics_out["rows"] = int(len(x))

        # helpful bounds for debugging
        if isinstance(x.index, pd.DatetimeIndex) and len(x.index) > 0:
            metrics_out["start_session_date"] = x.index.min().isoformat()
            metrics_out["end_session_date"] = x.index.max().isoformat()

        self.storage.write_json(metrics_out, metrics_key)

        # -----------------------------
        # 3) Meta json (latest) - richer info
        # -----------------------------
        meta_key = self.paths.performance_ts_meta(strategy=strategy, universe=universe)

        meta_out = {
            "generated_at_utc": now_iso,
            "strategy": strategy,
            "universe": universe,
            "ts_key": ts_key,
            "metrics_key": metrics_key,
            "shape": {"rows": int(x.shape[0]), "cols": int(x.shape[1])},
            "columns": list(map(str, x.columns)),
        }

        # index diagnostics
        if isinstance(x.index, pd.DatetimeIndex):
            meta_out["index"] = {
                "name": x.index.name,
                "tz": str(x.index.tz) if x.index.tz is not None else None,
                "dtype": str(x.index.dtype),
                "min": x.index.min().isoformat() if len(x.index) else None,
                "max": x.index.max().isoformat() if len(x.index) else None,
            }
        else:
            meta_out["index"] = {"name": getattr(x.index, "name", None), "dtype": str(x.index.dtype)}

        self.storage.write_json(meta_out, meta_key)


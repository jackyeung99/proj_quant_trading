# src/qbt/storage/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional


def _clean(x: str) -> str:
    """
    Make a string safe for partition-style paths (strategy=..., etc.).
    Keep it predictable across local FS + S3.
    """
    return (
        str(x).strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("=", "-")
    )


def _p(*parts: str) -> str:
    """Join POSIX key parts safely."""
    return str(PurePosixPath(*parts))


@dataclass(frozen=True)
class StoragePaths:
    """
    Key layout for QBT artifacts.

    Notes
    -----
    - All functions return *keys* (POSIX-like strings), not local filesystem Paths.
    - Storage backend maps keys -> local paths or S3 URIs.
    - Partition style: key=val folders for query-friendly datasets.
    """
    # top-level namespaces (inside storage root)
    results: str = "results"
    bronze: str = "data/bronze"
    silver: str = "data/silver"
    gold: str = "data/gold"
    state: str = "data/_state"

    # artifacts
    artifacts: str = "artifacts"
    models_dir: str = "artifacts/models"
    live_dir: str = "artifacts/live"              # weights + signals
    execution_dir: str = "artifacts/execution"    # locks + last_exec
    orders_dir: str = "artifacts/orders"          # planned orders (optional but useful)
    trades_dir: str = "artifacts/trades"          # executed trade ledger

    # ---------------------------------------------------------------------
    # Data construction pipeline
    # ---------------------------------------------------------------------
    # ----- bronze keys -----
    def bronze_bars_dir(self, *, freq: str, ticker: str) -> str:
        return _p(self.bronze, f"freq={_clean(freq)}", f"ticker={_clean(ticker)}")

    def bronze_bars_key(self, *, freq: str, ticker: str) -> str:
        return _p(self.bronze_bars_dir(freq=freq, ticker=ticker), "bars.parquet")

    def bronze_bars_state_key(self, *, freq: str, ticker: str) -> str:
        return _p(self.bronze_bars_dir(freq=freq, ticker=ticker), "state.json")

    # ----- silver keys -----
    def silver_bars_key(self, *, freq: str, ticker: str) -> str:
        return _p(self.silver, f"freq={_clean(freq)}", f"ticker={_clean(ticker)}", "bars.parquet")

    # ----- gold keys -----
    def gold_table_key(self, *, freq: str, tag: str = "default") -> str:
        return _p(self.gold, f"freq={_clean(freq)}", f"tag={_clean(tag)}", "table.parquet")

    def gold_manifest_key(self, *, freq: str, tag: str = "default") -> str:
        return _p(self.gold, f"freq={_clean(freq)}", f"tag={_clean(tag)}", "_manifest.json")

    # ----- state keys -----
    def source_state_key(self, *, source: str, dataset: str) -> str:
        return _p(self.state, _clean(source), f"{_clean(dataset)}.json")

    # ---------------------------------------------------------------------
    # Backtesting (global + per-run)
    # ---------------------------------------------------------------------
    def runs_key(self) -> str:
        return _p(self.results, "runs.parquet")

    def metrics_key(self) -> str:
        return _p(self.results, "metrics.parquet")

    def run_meta_key(self, run_id: str) -> str:
        return _p(self.results, "runs", f"run_id={_clean(run_id)}", "meta.json")

    def run_timeseries_key(self, strategy: str, universe: str, run_id: str) -> str:
        base = PurePosixPath(self.results) / "timeseries"
        p = (
            base
            / f"strategy={_clean(strategy)}"
            / f"universe={_clean(universe)}"
            / f"run_id={_clean(run_id)}"
        )
        return str(p / "timeseries.parquet")

    # ---------------------------------------------------------------------
    # Live artifacts: models + weights + execution guards + orders + trades
    # ---------------------------------------------------------------------
    # ----- models -----
    def model_key(self, strategy: str, universe: str = "default", tag: str = "latest") -> str:
        return _p(self.models_dir, f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", f"tag={_clean(tag)}", "model.pkl")

    def model_meta_key(self, strategy: str, universe: str = "default", tag: str = "latest") -> str:
        return _p(self.models_dir, f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", f"tag={_clean(tag)}", "meta.json")

    # ----- weights (signal output) -----
    def weights_latest_key(self, strategy: str, universe: str) -> str:
        return _p(self.live_dir, "weights", f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "latest.parquet")

    def weights_snapshot_key(self, strategy: str, universe: str, asof: str) -> str:
        return _p(self.live_dir, "weights", f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "snapshots", f"asof={_clean(asof)}.parquet")

    # ----- execution guards -----
    def exec_lock_key(self, strategy: str, universe: str) -> str:
        return _p(self.execution_dir, f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "lock.json")

    def last_exec_key(self, strategy: str, universe: str) -> str:
        return _p(self.execution_dir, f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "last_exec.json")

    # ----- planned orders (optional but useful) -----
    def orders_root(self, strategy: str, universe: str) -> str:
        return _p(self.orders_dir, f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "batches")

    def orders_batch_key(self, strategy: str, universe: str, date: str, batch_id: str) -> str:
        return _p(self.orders_root(strategy, universe), f"date={_clean(date)}", f"{_clean(batch_id)}.parquet")

    # ----- trades ledger (executed) -----
    def trades_root(self, strategy: str, universe: str) -> str:
        return _p(self.trades_dir, f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "batches")

    def trades_batch_key(self, strategy: str, universe: str, date: str, batch_id: str) -> str:
        return _p(self.trades_root(strategy, universe), f"date={_clean(date)}", f"{_clean(batch_id)}.parquet")

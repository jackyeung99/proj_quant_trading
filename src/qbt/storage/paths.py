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
    bronze: str = "data/bronze"
    silver: str = "data/silver"
    gold: str = "data/gold"
    state: str = "data/_state"

    # artifacts
    artifacts: str = "artifacts"
    results: str = "artifacts/backtesting_results"
    live_dir: str = "artifacts/live"              # weights + signals
    models_dir: str = "artifacts/live/models"
    weights_dir: str = "artifacts/live/weights"
    performance_dir: str = "artifacts/live/performance"
    execution_dir: str = "artifacts/live/execution"    # locks + last_exec
    orders_dir: str = "artifacts/live/orders"          # planned orders (optional but useful)
    trades_dir: str = "artifacts/live/trades"          # executed trade ledger

    # ---------------------------------------------------------------------
    # Data construction pipeline
    # ---------------------------------------------------------------------
    # ----- bronze keys -----
    def bronze_bars_dir(self, *, freq: str, ticker: str) -> str:
        return _p(self.bronze, f"freq={_clean(freq)}", f"symbol={_clean(ticker)}")

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
    def model_latest_prefix(self, strategy: str, universe: str) -> str:
        return f"{self.models_dir}/strategy={strategy}/universe={universe}/latest"

    def model_snapshot_prefix(self, strategy: str, universe: str) -> str:
        return f"{self.models_dir}/strategy={strategy}/universe={universe}/snapshots"
    

    def model_latest_pkl_key(self, strategy, universe):
        return f"{self.model_latest_prefix(strategy, universe)}/model.pkl"

    def model_latest_meta_key(self, strategy, universe):
        return f"{self.model_latest_prefix(strategy, universe)}/meta.json"

    def model_latest_params_key(self, strategy, universe):
        return f"{self.model_latest_prefix(strategy, universe)}/params.json"

    def model_snapshot_pkl_key(self, strategy, universe, trained_at):
        return f"{self.model_snapshot_prefix(strategy, universe)}/asof={_clean(trained_at)}/model.pkl"

    def model_snapshot_meta_key(self, strategy, universe, trained_at):
        return f"{self.model_snapshot_prefix(strategy, universe)}/asof={_clean(trained_at)}/meta.json"

    def model_snapshot_params_key(self, strategy, universe, trained_at):
        return f"{self.model_snapshot_prefix(strategy, universe)}/asof={_clean(trained_at)}/params.json"

    # ----- weights (signal output) -----
    def weights_latest_key(self, strategy: str, universe: str) -> str:
        return _p(self.weights_dir, f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "latest.parquet")

    def weights_snapshot_key(self, strategy: str, universe: str, asof: str) -> str:
        return _p(self.weights_dir, f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "snapshots", f"asof={_clean(asof)}.parquet")
    
    def weights_snapshots_prefix(self, strategy: str, universe: str):
         return _p(self.weights_dir, f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "snapshots")

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



    # ----- performance  -----

    def performance_ts(self, strategy, universe ) -> str:
        return _p(self.performance_dir,  f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "portfolio_timeseries.parquet")
    

    def performance_metrics(self, strategy, universe ) -> str:
        return _p(self.performance_dir,  f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "portfolio_metrics.json")
    
    def performance_ts_meta(self, strategy, universe ) -> str:
        return _p(self.performance_dir,  f"strategy={_clean(strategy)}", f"universe={_clean(universe)}", "meta.json")
    

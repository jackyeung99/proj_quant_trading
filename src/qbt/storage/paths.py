# src/qbt/storage/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


def _clean(x: str) -> str:
    """
    Make a string safe for partition-style paths (strategy=..., etc.).
    Keep it simple and predictable.
    """
    return (
        str(x).strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("=", "-")
    )


@dataclass(frozen=True)
class StoragePaths:
    """
    Key layout for result artifacts.

    IMPORTANT:
    - This returns *keys* (POSIX-like strings), not local filesystem Paths.
    - Your Storage backend decides how keys map to Local paths or S3 URIs.
    """
    results: str = "results"  # key prefix inside storage
    bronze: str = "data/bronze"
    silver: str = "data/silver"
    gold: str = "data/gold"
    state: str = "data/_state"


    # ----- bronze keys -----
    def bronze_bars_key(self, *, freq: str, ticker: str) -> str:
        return f"{self.bronze}/freq={freq}/ticker={ticker}/bars.parquet"

    # ----- silver keys -----
    def silver_bars_key(self, *, freq: str, ticker: str) -> str:
        return f"{self.silver}/freq={freq}/ticker={ticker}/bars.parquet"

    # ----- gold keys -----
    def gold_model_key(self, *, universe: str, freq: str = "1D") -> str:
        return f"{self.gold}/model_table/universe={universe}/freq={freq}/model.parquet"

    def gold_manifest_key(self, *, universe: str, freq: str = "1D") -> str:
        return f"{self.gold}/model_table/universe={universe}/freq={freq}/_manifest.json"

    # ----- state keys -----
    def source_state_key(self, *, source: str, dataset: str) -> str:
        return f"{self.state}/{source}/{dataset}.json"
    

    # ---- global artifacts ----
    def runs_key(self) -> str:
        return str(PurePosixPath(self.results) / "runs.parquet")

    def metrics_key(self) -> str:
        return str(PurePosixPath(self.results) / "metrics.parquet")


    # ---- per-run artifacts ----
    def run_meta_key(self, run_id: str) -> str:
        return str(PurePosixPath(self.results) / "runs" / _clean(run_id) / "meta.json")

    def run_timeseries_key(self, strategy: str, universe: str, run_id: str) -> str:
        """
        One file per run (MVP). Partition-style folders help filtering/query later.
        """
        base = PurePosixPath(self.results) / "timeseries"
        p = (
            base
            / f"strategy={_clean(strategy)}"
            / f"universe={_clean(universe)}"
            / f"run_id={_clean(run_id)}"
        )
        return str(p / "timeseries.parquet")

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
    root: str = "results"  # key prefix inside storage

    # ---- global artifacts ----
    def runs_key(self) -> str:
        return str(PurePosixPath(self.root) / "runs.parquet")

    def metrics_key(self) -> str:
        return str(PurePosixPath(self.root) / "metrics.parquet")

    # ---- per-run artifacts ----
    def run_meta_key(self, run_id: str) -> str:
        return str(PurePosixPath(self.root) / "runs" / _clean(run_id) / "meta.json")

    def run_timeseries_key(self, strategy: str, universe: str, run_id: str) -> str:
        """
        One file per run (MVP). Partition-style folders help filtering/query later.
        """
        base = PurePosixPath(self.root) / "timeseries"
        p = (
            base
            / f"strategy={_clean(strategy)}"
            / f"universe={_clean(universe)}"
            / f"run_id={_clean(run_id)}"
        )
        return str(p / "timeseries.parquet")

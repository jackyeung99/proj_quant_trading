from __future__ import annotations

from typing import Any, Dict
from pathlib import PurePosixPath

from datetime import datetime, timezone
import pandas as pd

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths
from qbt.core.logging import get_logger
from qbt.data.canonical import canonicalize_bars, canonicalize_macro



logging = get_logger(__name__)


def canonicalize_all(storage: Storage, paths: StoragePaths, cfg: dict) -> dict:
    sources_cfg = cfg.get("sources", {}) or {}
    silver_cfg = cfg.get("silver", {}) or {}
    if not silver_cfg.get("enabled", True):
        return {"canonicalize_results": {}, "skipped": True}

    results: Dict[str, Any] = {}

    for source_name, sc in sources_cfg.items():
        if not sc.get("enabled", False):
            continue

        interval = sc.get("interval")
        symbols = sc.get("symbols", []) or []
    
        is_macro = source_name.lower().startswith("macro")

        if is_macro:
            # ---- macro canonicalization ----
            out_stats = []
            for sym in symbols:
                bronze_key = paths.bronze_macro_key(dataset=source_name, freq=interval, symbol=sym)
                silver_key = paths.silver_macro_key(dataset=source_name, freq=interval, symbol=sym)

                if not storage.exists(bronze_key):
                    raise FileNotFoundError(bronze_key)

                raw = storage.read_parquet(bronze_key)

                print(raw)

                clean = canonicalize_macro(
                    raw,
                    date_col="date",          # adjust to your macro schema
                    value_col="value",
                )

                storage.write_parquet(clean, silver_key)
                out_stats.append({"symbol": sym, "rows": int(len(clean))})

            results[source_name] = {"kind": "macro", "interval": interval, "n": len(out_stats), "items": out_stats}

        else:
            # ---- bar canonicalization ----
            out_stats = []
            for sym in symbols:
                bronze_key = paths.bronze_bars_key(dataset=source_name, freq=interval, ticker=sym)
                silver_key = paths.silver_bars_key(dataset=source_name, freq=interval, ticker=sym)

                if not storage.exists(bronze_key):
                    raise FileNotFoundError(bronze_key)

                raw = storage.read_parquet(bronze_key)

                # Choose required columns based on daily vs intraday (or read from cfg)
                require_cols = ("open", "high", "low", "close") if "daily" in source_name else ("open", "close")

                clean = canonicalize_bars(
                    raw,
                    timestamp_col="timestamp",
                    require_cols=require_cols,
                    tz_out="UTC",
                )

                storage.write_parquet(clean, silver_key)
                out_stats.append({"ticker": sym, "rows": int(len(clean))})

            results[source_name] = {"kind": "bars", "interval": interval, "n": len(out_stats), "items": out_stats}


    return {"canonicalize_results": results}

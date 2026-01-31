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
    logging.info("Starting canonicalize_all")

    sources_cfg = cfg.get("sources", {}) or {}
    silver_cfg = cfg.get("silver", {}) or {}

    if not silver_cfg.get("enabled", True):
        logging.info("Silver layer disabled; skipping canonicalization")
        return {"canonicalize_results": {}, "skipped": True}

    results: Dict[str, Any] = {}

    for source_name, sc in sources_cfg.items():
        if not sc.get("enabled", False):
            logging.debug("Source disabled, skipping: %s", source_name)
            continue

        interval = sc.get("interval")
        symbols = sc.get("symbols", []) or []
        is_macro = sc.get("provider") == "fred"

        logging.info(
            "Canonicalizing source=%s kind=%s interval=%s symbols=%d",
            source_name,
            "macro" if is_macro else "bars",
            interval,
            len(symbols),
        )

        if is_macro:
            # ---- macro canonicalization ----
            out_stats = []
            for sym in symbols:
                bronze_key = paths.bronze_bars_key(freq=interval, ticker=sym)
                silver_key = paths.silver_bars_key(freq=interval, ticker=sym)

                logging.debug("Macro %s: reading %s", sym, bronze_key)

                if not storage.exists(bronze_key):
                    logging.error("Missing bronze macro key: %s", bronze_key)
                    raise FileNotFoundError(bronze_key)

                raw = storage.read_parquet(bronze_key)
                n_raw = 0 if raw is None else len(raw)

                clean = canonicalize_macro(
                    raw,
                    timestamp_col="timestamp",
                    value_col="value",
                )

                n_clean = len(clean)
                logging.info(
                    "Macro %s: rows in=%d rows out=%d -> %s",
                    sym,
                    n_raw,
                    n_clean,
                    silver_key,
                )

                storage.write_parquet(clean, silver_key)
                out_stats.append({"symbol": sym, "rows": int(n_clean)})

            results[source_name] = {
                "kind": "macro",
                "interval": interval,
                "n": len(out_stats),
                "items": out_stats,
            }

        else:
            # ---- bar canonicalization ----
            out_stats = []
            for sym in symbols:
                bronze_key = paths.bronze_bars_key(freq=interval, ticker=sym)
                silver_key = paths.silver_bars_key(freq=interval, ticker=sym)

                logging.debug("Bars %s: reading %s", sym, bronze_key)

                if not storage.exists(bronze_key):
                    logging.error("Missing bronze bars key: %s", bronze_key)
                    raise FileNotFoundError(bronze_key)

                raw = storage.read_parquet(bronze_key)
                n_raw = 0 if raw is None else len(raw)

                require_cols = (
                    ("open", "high", "low", "close")
                    if "daily" in source_name
                    else ("open", "close")
                )

                clean = canonicalize_bars(
                    raw,
                    timestamp_col="timestamp",
                    require_cols=require_cols,
                    tz_out="UTC",
                )

                n_clean = len(clean)
                logging.info(
                    "Bars %s: rows in=%d rows out=%d -> %s",
                    sym,
                    n_raw,
                    n_clean,
                    silver_key,
                )

                storage.write_parquet(clean, silver_key)
                out_stats.append({"ticker": sym, "rows": int(n_clean)})

            results[source_name] = {
                "kind": "bars",
                "interval": interval,
                "n": len(out_stats),
                "items": out_stats,
            }

    logging.info("Finished canonicalize_all")
    return {"canonicalize_results": results}

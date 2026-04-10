from __future__ import annotations

from typing import Any, Dict
from pathlib import PurePosixPath

import pandas as pd

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths
from qbt.data.sources.source_registry import create_source, available_sources
from qbt.data.state import get_last_available_date, compute_fetch_window, update_state
from qbt.data.merge import merge_and_dedup

from qbt.core.logging import get_logger
logging = get_logger(__name__)



def ingest_one_source(
    storage: Storage,
    paths: StoragePaths,
    dataset_cfg: dict,
    source_cfg: dict,
    dataset_name: str,
    src: Any,
) -> dict:
  
    mode = dataset_cfg.get("mode")
    lookback = dataset_cfg.get("lookback_days")
    start_override = dataset_cfg.get("start_override")
    end_override = dataset_cfg.get("end_override")

    provider = source_cfg.get("connection")
    if not provider:
        raise ValueError(f"{dataset_name}: missing provider")

    results_per_ticker: Dict[str, Any] = {}

    logging.info(
        "INGEST %s | provider=%s | ingestion_mode=%s",
        dataset_name, provider, mode,
    )

    symbols = source_cfg.get("symbols", []) or []
    if not symbols:
        logging.warning(
            "INGEST %s | provider=%s | no symbols configured",
            dataset_name, provider,
        )

    last_fetch_start = None
    last_fetch_end = None

    for ticker in symbols:
        ticker = str(ticker).strip()
        freq = source_cfg.get("interval")

        if not ticker:
            continue

        store_key = paths.bronze_bars_key(freq=freq, ticker=ticker)
        state_key = paths.bronze_bars_state_key(freq=freq, ticker=ticker)

        last_date = get_last_available_date(storage, state_key)

        fetch_start, fetch_end = compute_fetch_window(
            last_date=last_date,
            lookback_days=lookback,
            mode=mode,
            start_override=start_override,
            end_override=end_override,
        )

        last_fetch_start = fetch_start
        last_fetch_end = fetch_end

        logging.info(
            "SOURCE %s | dataset=%s | ticker=%s | freq=%s | fetching %s -> %s",
            provider, dataset_name, ticker, freq, fetch_start, fetch_end,
        )

        new_df = src.fetch(ticker=ticker, start=fetch_start, end=fetch_end)
        new_df = src.standardize(new_df)
        src.validate(new_df)

        if storage.exists(store_key):
            old_df = storage.read_parquet(store_key)
            merged = merge_and_dedup(old_df, new_df)
        else:
            merged = new_df

        storage.write_parquet(merged, store_key)

        update_state(
            storage,
            state_key,
            merged,
            pull_start=fetch_start,
            pull_end=fetch_end,
            meta={"ingestion": dataset_cfg, "source": source_cfg},
        )

        results_per_ticker[ticker] = {
            "store_key": store_key,
            "rows_new": int(len(new_df)),
        }

    return {
        "dataset": dataset_name,
        "provider": provider,
        "fetch_window": (last_fetch_start, last_fetch_end),
        "tickers": results_per_ticker,
    }


def ingest_all_sources(
    storage: Storage,
    paths: StoragePaths,
    dataset_cfg: dict,
    sources: dict[str, Any],
) -> dict:
    results: Dict[str, Any] = {}

    for source_cfg in dataset_cfg["inputs"]:
      
        if not isinstance(source_cfg, dict):
            continue

        source_name = str(source_cfg.get("name"))
        connection = source_cfg.get("connection", None)
 
        if not connection:
            raise ValueError(f"{source_name}: missing connection")

        if connection not in sources:
            raise KeyError(
                f"{source_name}: connection {connection!r} not found in injected sources. "
                f"Available: {sorted(sources.keys())}"
            )

        src = sources[connection]

        results[source_name] = ingest_one_source(
            storage=storage,
            paths=paths,
            dataset_cfg=dataset_cfg,
            source_cfg=source_cfg,
            dataset_name=source_name,
            src=src,
        )

    return {"ingestion_results": results}
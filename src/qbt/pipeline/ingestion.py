from __future__ import annotations

from typing import Any, Dict
from pathlib import PurePosixPath

import pandas as pd

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths
from qbt.data.sources.source_registry import create_source, available_sources
from qbt.data.state import get_last_available_date, compute_fetch_window, update_state
from qbt.core.logging import get_logger

logging = get_logger(__name__)



def ingest_one_source(storage: Storage, paths: StoragePaths, cfg: dict, dataset_name: str, source_cfg: dict) -> dict:
    ingestion_cfg = cfg.get("ingestion", {}) or {}

    provider = source_cfg.get("provider")
    if not provider:
        raise ValueError(f"{dataset_name}: missing provider")

    mode = source_cfg.get('mode')
    lookback = source_cfg.get('lookback_days')
    start_override = source_cfg.get('start', None)
    end_override = source_cfg.get('end', None)


    # cfg-only: pass the *whole* source_cfg into the source
    # (this is where api_key/api_secret can live if you merged them in the entrypoint)
    src = create_source(provider, cfg=source_cfg)

    results_per_ticker: Dict[str, Any] = {}

    logging.info(
        "INGEST %s | provider=%s |ingestion mode = %s",
        dataset_name, provider, mode
    )

    symbols = source_cfg.get("symbols", []) or []
    if not symbols:
        logging.warning("INGEST %s | provider=%s | no symbols configured", dataset_name, provider)

    for ticker in symbols:

        ticker = str(ticker).strip()
        freq = source_cfg.get("interval")

        if not ticker:
            continue

        store_key = paths.bronze_bars_key(ticker=ticker, freq=freq)


        last_date = get_last_available_date(storage, store_key)

        fetch_start, fetch_end = compute_fetch_window(
                                                    last_date=last_date,
                                                    lookback_days=lookback,
                                                    mode=mode,
                                                    start_override=start_override,
                                                    end_override= end_override
                                                )


        logging.info(
            "SOURCE %s | dataset=%s | ticker=%s |freq= %s | fetching %s -> %s",
            provider, dataset_name, ticker, freq, fetch_start, fetch_end
        )

        # if a source needs ticker in cfg instead of as arg, it can read it from cfg
        # but this keeps a clean canonical signature.
        df = src.fetch(ticker=ticker, start=fetch_start, end=fetch_end)
        df = src.standardize(df)
        src.validate(df)

    
        storage.write_parquet(df, store_key)

        logging.debug(
            "SOURCE %s | dataset=%s | ticker=%s | rows=%d | stored_at=%s",
            provider, dataset_name, ticker, len(df), store_key
        )

        update_state(storage, store_key, df, fetch_start, fetch_end)

        results_per_ticker[ticker] = {"store_key": store_key, "rows_new": int(len(df))}

    return {
        "dataset": dataset_name,
        "provider": provider,
        "fetch_window": (fetch_start, fetch_end),
        "tickers": results_per_ticker,
    }


def ingest(storage: Storage, paths:StoragePaths, cfg: dict) -> dict:
    logging.debug("Using Storage %s base_dir=%s", storage, getattr(storage, "base_dir", None))
    logging.debug("Available Sources: %s", available_sources())

    sources_cfg = cfg.get("sources", {}) or {}
    results: Dict[str, Any] = {}

    for dataset_name, source_cfg in sources_cfg.items():
        if not isinstance(source_cfg, dict) or not source_cfg.get("enabled", True):
            continue

        results[str(dataset_name)] = ingest_one_source(
            storage=storage,
            paths=paths,
            cfg=cfg,
            dataset_name=str(dataset_name),
            source_cfg=source_cfg,
        )

    return {"ingestion_results": results}

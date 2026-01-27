from __future__ import annotations

from typing import Any, Dict
from pathlib import PurePosixPath

import pandas as pd

from qbt.storage.storage import Storage
from qbt.data.sources.source_registry import create_source, available_sources
from qbt.core.logging import get_logger

logging = get_logger(__name__)


def _parse_window(ingestion_cfg: dict, source_cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Allow per-source override:
      source_cfg.start / source_cfg.end override ingestion.start/end
    Use half-open [start, end) by bumping date-only end +1 day.
    """
    start_raw = source_cfg.get("start", ingestion_cfg.get("start"))
    end_raw = source_cfg.get("end", ingestion_cfg.get("end"))

    if start_raw is None or end_raw is None:
        raise ValueError("Missing start/end in ingestion config")

    start = pd.to_datetime(start_raw, utc=True)
    end = pd.to_datetime(end_raw, utc=True)

    if end == end.normalize():
        end = end + pd.Timedelta(days=1)

    return start, end


def _build_store_key(cfg: dict, source_cfg: dict, symbol: str) -> str:
    """
    prefix: cfg.paths.prefix (optional)
    partitioning: freq=<interval|timeframe>/ticker=<symbol>/bars.parquet
    """
    prefix = cfg.get("paths", {}).get("prefix", "").strip("/")

    freq = source_cfg.get("interval") or source_cfg.get("timeframe")
    if not freq:
        raise ValueError("source_cfg must contain 'interval' or 'timeframe'")

    symbol = str(symbol).strip()
    if not symbol:
        raise ValueError("symbol must be non-empty")

    key = PurePosixPath(prefix) / f"freq={freq}" / f"ticker={symbol}" / "bars.parquet"
    return str(key)


def ingest_one_source(storage: Storage, cfg: dict, dataset_name: str, source_cfg: dict) -> dict:
    ingestion_cfg = cfg.get("ingestion", {}) or {}

    provider = source_cfg.get("provider")
    if not provider:
        raise ValueError(f"{dataset_name}: missing provider")

    fetch_start, fetch_end = _parse_window(ingestion_cfg, source_cfg)

    logging.info(
        "INGEST %s | provider=%s | %s -> %s",
        dataset_name, provider, fetch_start, fetch_end
    )

    # cfg-only: pass the *whole* source_cfg into the source
    # (this is where api_key/api_secret can live if you merged them in the entrypoint)
    src = create_source(provider, cfg=source_cfg)

    results_per_ticker: Dict[str, Any] = {}

    symbols = source_cfg.get("symbols", []) or []
    if not symbols:
        logging.warning("INGEST %s | provider=%s | no symbols configured", dataset_name, provider)

    for ticker in symbols:
        ticker = str(ticker).strip()
        if not ticker:
            continue

        store_key = _build_store_key(cfg, source_cfg, ticker)

        logging.info(
            "SOURCE %s | dataset=%s | ticker=%s | fetching %s -> %s",
            provider, dataset_name, ticker, fetch_start, fetch_end
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

        results_per_ticker[ticker] = {"store_key": store_key, "rows_new": int(len(df))}

    return {
        "dataset": dataset_name,
        "provider": provider,
        "fetch_window": (fetch_start, fetch_end),
        "tickers": results_per_ticker,
    }


def ingest(storage: Storage, cfg: dict) -> dict:
    logging.debug("Using Storage %s base_dir=%s", storage, getattr(storage, "base_dir", None))
    logging.debug("Available Sources: %s", available_sources())

    sources_cfg = cfg.get("sources", {}) or {}
    results: Dict[str, Any] = {}

    for dataset_name, source_cfg in sources_cfg.items():
        if not isinstance(source_cfg, dict) or not source_cfg.get("enabled", True):
            continue

        results[str(dataset_name)] = ingest_one_source(
            storage=storage,
            cfg=cfg,
            dataset_name=str(dataset_name),
            source_cfg=source_cfg,
        )

    return {"ingestion_results": results}

from __future__ import annotations

from typing import Any, Dict

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths
from qbt.config.specs import DatasetSpec, DatasetSourceSpec
from qbt.data.state import get_last_available_date, compute_fetch_window, update_state
from qbt.data.merge import merge_and_dedup

from qbt.core.logging import get_logger

logger = get_logger(__name__)


def ingest_one_source(
    storage: Storage,
    paths: StoragePaths,
    dataset: DatasetSpec,
    source_spec: DatasetSourceSpec,
    src: Any,
) -> dict:

    # optional fields if you later add them into DatasetSpec
    start_override = dataset.aggregation.get("start_override")
    end_override = dataset.aggregation.get("end_override")

    connection = source_spec.connection
    if not connection:
        raise ValueError(f"{source_spec.name}: missing connection")

    results_per_ticker: Dict[str, Any] = {}

    logger.info(
        "INGEST %s | connection=%s | ingestion_mode=%s",
        source_spec.name,
        connection,
        dataset.mode,
    )

    symbols = source_spec.symbols or []
    if not symbols:
        logger.warning(
            "INGEST %s | connection=%s | no symbols configured",
            source_spec.name,
            connection,
        )

    last_fetch_start = None
    last_fetch_end = None

    for ticker in symbols:
        ticker = str(ticker).strip()
        if not ticker:
            continue

        freq = source_spec.interval

        store_key = paths.bronze_bars_key(freq=freq, ticker=ticker)
        state_key = paths.bronze_bars_state_key(freq=freq, ticker=ticker)

        last_date = get_last_available_date(storage, state_key)

        fetch_start, fetch_end = compute_fetch_window(
            last_date=last_date,
            lookback_days=dataset.lookback_days,
            mode=dataset.mode,
            start_override=start_override,
            end_override=end_override,
        )

        last_fetch_start = fetch_start
        last_fetch_end = fetch_end

        logger.info(
            "SOURCE %s | dataset=%s | ticker=%s | freq=%s | fetching %s -> %s",
            connection,
            source_spec.name,
            ticker,
            freq,
            fetch_start,
            fetch_end,
        )

        new_df = src.fetch(
            ticker=ticker,
            start=fetch_start,
            end=fetch_end,
            interval=source_spec.interval,
            **source_spec.params,
        )

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
            meta={
                "dataset_name": dataset.name,
                "dataset_mode": dataset.mode,
                "source_name": source_spec.name,
                "connection": source_spec.connection,
                "interval": source_spec.interval,
                "params": source_spec.params,
            },
        )

        results_per_ticker[ticker] = {
            "store_key": store_key,
            "rows_new": int(len(new_df)),
        }

    return {
        "dataset": dataset.name,
        "source_name": source_spec.name,
        "connection": connection,
        "fetch_window": (last_fetch_start, last_fetch_end),
        "tickers": results_per_ticker,
    }


def ingest_all_sources(
    storage: Storage,
    paths: StoragePaths,
    dataset: DatasetSpec,
    sources: dict[str, Any],
) -> dict:
    results: Dict[str, Any] = {}

    for source_spec in dataset.sources:
        connection = source_spec.connection

        if not connection:
            raise ValueError(f"{source_spec.name}: missing connection")

        if connection not in sources:
            raise KeyError(
                f"{source_spec.name}: connection {connection!r} not found in injected sources. "
                f"Available: {sorted(sources.keys())}"
            )

        src = sources[connection]

        results[source_spec.name] = ingest_one_source(
            storage=storage,
            paths=paths,
            dataset=dataset,
            source_spec=source_spec,
            src=src,
        )

    return {"ingestion_results": results}
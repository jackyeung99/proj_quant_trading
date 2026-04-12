from __future__ import annotations

from typing import Any, Dict

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths
from qbt.core.logging import get_logger
from qbt.data.canonical import canonicalize_bars, canonicalize_macro
from qbt.config.specs import DatasetSpec, DatasetSourceSpec


logger = get_logger(__name__)


def _canonicalize_one_source(
    storage: Storage,
    paths: StoragePaths,
    source_spec: DatasetSourceSpec,
) -> dict[str, Any]:
    source_name = source_spec.name
    interval = source_spec.interval
    symbols = source_spec.symbols or []
    is_macro = source_spec.kind == "macro"

    logger.info(
        "Canonicalizing source=%s kind=%s interval=%s symbols=%d",
        source_name,
        source_spec.kind,
        interval,
        len(symbols),
    )

    out_stats: list[dict[str, Any]] = []

    for sym in symbols:
        bronze_key = paths.bronze_bars_key(freq=interval, ticker=sym)
        silver_key = paths.silver_bars_key(freq=interval, ticker=sym)

        logger.debug(
            "%s %s: reading %s",
            "Macro" if is_macro else "Bars",
            sym,
            bronze_key,
        )

        if not storage.exists(bronze_key):
            logger.error("Missing bronze key: %s", bronze_key)
            raise FileNotFoundError(bronze_key)

        raw = storage.read_parquet(bronze_key)
        n_raw = 0 if raw is None else len(raw)

        if is_macro:
            clean = canonicalize_macro(
                raw,
                timestamp_col="timestamp",
                value_col="value",
            )
        else:
            require_cols = (
                ("open", "high", "low", "close")
                if "daily" in source_name.lower()
                else ("open", "close")
            )

            clean = canonicalize_bars(
                raw,
                timestamp_col="timestamp",
                require_cols=require_cols,
                tz_out="UTC",
            )

        n_clean = len(clean)

        logger.info(
            "%s %s: rows in=%d rows out=%d -> %s",
            "Macro" if is_macro else "Bars",
            sym,
            n_raw,
            n_clean,
            silver_key,
        )

        storage.write_parquet(clean, silver_key)

        out_stats.append(
            {
                "symbol" if is_macro else "ticker": sym,
                "rows": int(n_clean),
            }
        )

    return {
        "kind": source_spec.kind,
        "interval": interval,
        "n": len(out_stats),
        "items": out_stats,
    }


def canonicalize_all(
    storage: Storage,
    paths: StoragePaths,
    dataset: DatasetSpec,
) -> dict[str, Any]:
    logger.info("Starting canonicalize_all for dataset=%s", dataset.name)

    results: Dict[str, Any] = {}

    for source_spec in dataset.sources:
        results[source_spec.name] = _canonicalize_one_source(
            storage=storage,
            paths=paths,
            source_spec=source_spec,
        )

    logger.info("Finished canonicalize_all for dataset=%s", dataset.name)
    return {"canonicalize_results": results}
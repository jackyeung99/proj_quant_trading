from __future__ import annotations

from typing import Any, Dict
import pandas as pd 

from qbt.data.sources import yfin, fred, weather
from qbt.storage.storage import Storage

from datetime import datetime, timezone, timedelta

from qbt.core.logging import get_logger

logging = get_logger(__name__)


def fetch_by_source(source_name: str, fetch_start, fetch_end, source_cfg: dict):
    """
    Only fetch differs per source.
    - equities: yfin.fetch(start, end, tickers=..., interval=..., ...)
    - macro:    fred.fetch(start, end, series=..., provider=..., ...)
    - weather:  weather.fetch(start, end, location=..., units=..., ...)
    Adjust the kwargs mappings to your real config keys.
    """
    if source_name == "equities":
        return yfin.fetch(
            tickers=source_cfg["symbols"],
            start=fetch_start,
            end=fetch_end,
            interval=source_cfg.get("interval", "1d"),
        )

    if source_name == "macro":
        return fred.fetch(
            series=source_cfg["series"],
            start=fetch_start,
            end=fetch_end
        )

    if source_name == "weather":
        return weather.fetch(
            cities=source_cfg["locations"],
            variables=source_cfg["variables"],
            start=fetch_start,
            end=fetch_end
        )

    raise ValueError(f"Unknown source '{source_name}'")


def standardize_by_source(source_name: str, df,):
    if source_name == "equities":
        return yfin.standardize(df)
    if source_name == "macro":
        return fred.standardize(df)
    if source_name == "weather":
        return weather.standardize(df)
    raise ValueError(f"Unknown source '{source_name}'")


def validate_by_source(source_name: str, df) -> None:
    if source_name == "equities":
        return yfin.validate(df)
    if source_name == "macro":
        return fred.validate(df)
    if source_name == "weather":
        return weather.validate(df)
    raise ValueError(f"Unknown source '{source_name}'")


def ingest_one_source(
    storage: Storage,
    global_cfg: dict,
    base_ingest_cfg: dict,
    source_name: str,
    source_cfg: dict,
) -> dict:
    store_key = source_cfg["store_path"]



    fetch_start = pd.to_datetime(base_ingest_cfg.get("start"), utc=True) # parse
    fetch_end = pd.to_datetime(base_ingest_cfg.get("end"), utc=True)


    logging.info(
        "SOURCE %s | fetching data from %s to %s",
        source_name,
        fetch_start,
        fetch_end,
    )

    # # 3) fetch (only part that differs)
    new_df = fetch_by_source(source_name, fetch_start, fetch_end, source_cfg)

    # # 4) standardize + validate
    new_df = standardize_by_source(source_name, new_df)
    validate_by_source(source_name, new_df)

    # # 5) merge/write/state
    # if storage.exists(store_key):
    #     old_df = storage.read_parquet(store_key)
    #     merged = merge_and_dedup(old_df, new_df)
    # else:
    merged = new_df

    storage.write_parquet(merged, store_key)

    logging.info(
        "SOURCE %s | rows written %s ",
        source_name,
        int(len(new_df))
    )

    # update_state(
    #     storage,
    #     store_key,
    #     merged,
    #     pull_start=fetch_start,
    #     pull_end=fetch_end,
    #     meta={"ingestion": base_ingest_cfg, "source": source_cfg},
    # )

    last_ts = merged.index.max() if len(merged) else None
    return {
        "source": source_name,
        "store_key": store_key,
        "rows_written": int(len(new_df)),
        "last_timestamp": last_ts,
        "fetch_window": (fetch_start, fetch_end),
    }


def ingest(storage: Storage, global_cfg: dict, step_cfg: dict) -> dict:
    base_ingest_cfg = step_cfg["ingestion"]
    sources_cfg = step_cfg["sources"]

    results: Dict[str, Any] = {}


    # Explicit per-source calls
    eq_cfg = sources_cfg.get("equities_intra", {})
    if eq_cfg.get("enabled", True):
        results["equities_intra"] = ingest_one_source(storage, global_cfg, base_ingest_cfg, "equities", eq_cfg)

    eq_daily_cfg = sources_cfg.get("equities_daily", {})
    if eq_daily_cfg.get("enabled", True):
        results["equities_daily"] = ingest_one_source(storage, global_cfg, base_ingest_cfg, "equities", eq_daily_cfg)

    macro_cfg = sources_cfg.get("macro", {})
    if macro_cfg.get("enabled", True):
        results["macro"] = ingest_one_source(storage, global_cfg, base_ingest_cfg, "macro", macro_cfg)

    weather_cfg = sources_cfg.get("weather", {})
    if weather_cfg.get("enabled", True):
        results["weather"] = ingest_one_source(storage, global_cfg, base_ingest_cfg, "weather", weather_cfg)

    return {"ingestion_results": results}

from __future__ import annotations

from typing import Sequence

import pandas as pd

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths




def load_multi_asset_flat_long(
    storage: Storage,
    paths: StoragePaths,
    assets: Sequence[str],
    freq: str = "15Min",
    timestamp_col: str = "timestamp",
    fields: Sequence[str] = ("open", "high", "low", "close", "volume"),
    asset_col: str = "ticker",
    utc: bool = True,
) -> pd.DataFrame:
    """
    Load many assets from silver into one LONG DataFrame.

    Output columns: [timestamp_col, asset_col, <fields...>]
    """
    frames: list[pd.DataFrame] = []

    for a in assets:
        key = paths.silver_bars_key(freq=freq, ticker=a)
        df = storage.read_parquet(key=key)
        if df is None or len(df) == 0:
            continue

        # timestamp from column or index
        ts_raw = df[timestamp_col] if timestamp_col in df.columns else df.index
        ts = pd.to_datetime(ts_raw, errors="coerce", utc=utc)

        out = df.copy()
        out[timestamp_col] = ts
        out[asset_col] = a

        # ensure requested fields exist
        for c in fields:
            if c not in out.columns:
                out[c] = pd.NA

        out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

        keep = [timestamp_col, asset_col, *fields]
        frames.append(out[keep])

    if not frames:
        return pd.DataFrame(columns=[timestamp_col, asset_col, *fields])

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values([asset_col, timestamp_col])
    raw = raw.drop_duplicates(subset=[asset_col, timestamp_col], keep="last")
    return raw
from __future__ import annotations


import pandas as pd



def canonicalize_bars(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    tz: str = "UTC",
    require_cols: tuple[str, ...] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    out = df.copy()

    # timestamps
    if timestamp_col in out.columns:
        ts = pd.to_datetime(out[timestamp_col], errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(out.index, errors="coerce", utc=True)
        out = out.reset_index().rename(columns={"index": timestamp_col})

    out[timestamp_col] = ts
    out = out.dropna(subset=[timestamp_col])

    # required columns
    missing = [c for c in require_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # basic numeric coercion
    for c in require_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # optional volume
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    # sort + dedupe
    out = out.sort_values(timestamp_col)
    out = out.drop_duplicates(subset=[timestamp_col], keep="last")

    # drop rows where OHLC invalid
    out = out.dropna(subset=["open", "close"])

    return out


def canonicalize_macro(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    name_col: str = "name",
    value_col: str = "value",
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Canonicalize a macro time series table.

    Expected input (long):
        timestamp | name | value

    Output invariants:
    - timestamp tz-aware (UTC)
    - unique (name, timestamp)
    - sorted by (name, timestamp)
    - numeric value
    """

    if df is None:
        return pd.DataFrame(columns=[timestamp_col, name_col, value_col])

    out = df.copy()

    # --- ensure required columns exist ---
    for c in (timestamp_col, name_col, value_col):
        if c not in out.columns:
            out[c] = pd.NA

    # --- timestamp parsing ---
    out[timestamp_col] = pd.to_datetime(
        out[timestamp_col],
        errors="coerce",
        utc=True,
    )

    # --- name normalization ---
    out[name_col] = out[name_col].astype(str)

    # --- value normalization ---
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    # --- drop rows missing keys ---
    out = out.dropna(subset=[timestamp_col, name_col])

    # --- sort + dedupe ---
    out = (
        out.sort_values([name_col, timestamp_col])
           .drop_duplicates([name_col, timestamp_col], keep="last")
           .reset_index(drop=True)
    )

    return out[[timestamp_col, name_col, value_col]]
from __future__ import annotations


import pandas as pd



def canonicalize_bars(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    tz_out: str = "UTC",
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
    out_dt_col: str = "datetime",
    out_ticker_col: str = "ticker",
    out_close_col: str = "close",
) -> pd.DataFrame:
    """
    Canonicalize a macro long table to bars-like schema:

        datetime | ticker | close

    Invariants:
    - datetime tz-aware UTC
    - ticker string (stripped, upper)
    - close numeric (float)
    - unique (ticker, datetime), keep="last"
    - sorted by (ticker, datetime)
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[out_dt_col, out_ticker_col, out_close_col])

    # Fail fast if required columns missing (cleaner than silently creating NA columns)
    missing = [c for c in (timestamp_col, name_col, value_col) if c not in df.columns]
    if missing:
        raise ValueError(f"canonicalize_macro missing columns: {missing}")

    out = df[[timestamp_col, name_col, value_col]].copy()

    # datetime (UTC, tz-aware)
    out[out_dt_col] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")

    # ticker
    out[out_ticker_col] = (
        out[name_col]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # close
    out[out_close_col] = pd.to_numeric(out[value_col], errors="coerce")

    # drop bad keys
    out = out.dropna(subset=[out_dt_col, out_ticker_col])

    # sort + dedupe
    out = (
        out.sort_values([out_ticker_col, out_dt_col])
           .drop_duplicates([out_ticker_col, out_dt_col], keep="last")
           .reset_index(drop=True)
    )

    return out[[out_dt_col, out_ticker_col, out_close_col]]
from __future__ import annotations
import pandas as pd


def merge_and_dedup(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        return new_df.copy() if new_df is not None else pd.DataFrame()
    if new_df is None or new_df.empty:
        return old_df.copy()

    old = old_df.copy()
    new = new_df.copy()

    # --- ensure timestamp column exists ---
    if timestamp_col not in old.columns or timestamp_col not in new.columns:
        raise ValueError(f"Missing '{timestamp_col}' column in one of the DataFrames")

    # --- parse timestamps (UTC-safe) ---
    old[timestamp_col] = pd.to_datetime(old[timestamp_col], utc=True, errors="coerce")
    new[timestamp_col] = pd.to_datetime(new[timestamp_col], utc=True, errors="coerce")

    # --- drop invalid timestamps ---
    old = old.dropna(subset=[timestamp_col])
    new = new.dropna(subset=[timestamp_col])

    # --- dedupe within each (keep last) ---
    old = old.sort_values(timestamp_col).drop_duplicates(
        subset=[timestamp_col], keep="last"
    )
    new = new.sort_values(timestamp_col).drop_duplicates(
        subset=[timestamp_col], keep="last"
    )

    # --- concat; new wins on conflicts ---
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.sort_values(timestamp_col).drop_duplicates(
        subset=[timestamp_col], keep="last"
    )

    return merged.reset_index(drop=True)
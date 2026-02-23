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

def canonical_session_index(df: pd.DataFrame, *, name: str = "session_date") -> pd.DataFrame:
    """
    Force index to be the canonical daily join key:
      - DatetimeIndex
      - tz-naive
      - normalized to midnight
      - sorted
      - de-duped (keep last)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.DatetimeIndex(pd.to_datetime(x.index, errors="coerce"))

    idx = x.index

    # If tz-aware, convert to UTC then drop tz
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)

    idx = idx.normalize()

    x.index = pd.DatetimeIndex(idx, name=name)
    x = x.sort_index()
    x = x[~x.index.duplicated(keep="last")]
    return x

def _merge_asof_left(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    """
    Asof-merge on index (expects both indexed by session_date).
    Backward direction (last known value in force).
    """

    if not isinstance(left.index, pd.DatetimeIndex):
        raise ValueError("Left must be indexed by DatetimeIndex (session_date).")

    if not isinstance(right.index, pd.DatetimeIndex):
        raise ValueError("Right must be indexed by DatetimeIndex (session_date).")

    l = left.sort_index()
    r = right.sort_index()

    return pd.merge_asof(
        l,
        r,
        left_index=True,
        right_index=True,
        direction="backward",
    )

def fill_in_force(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic forward-fill policy:

      - *_weight            -> numeric ffill then 0
      - *_utc               -> datetime (tz-aware UTC) ffill
      - id-like columns     -> string ffill
      - numeric parameters  -> numeric ffill
      - string parameters   -> string ffill
      - portfolio_value/ret -> leave untouched
    """
    x = df.copy()

    # Explicit categories
    id_like = {"config_hash", "snapshot_id", "model_snapshot_id", "market_tz"}
    no_fill = {"portfolio_value", "ret"}

    for c in x.columns:
        if c in no_fill:
            continue

        s = x[c]

        # ----------------------------------------
        # 1) Weight columns
        # ----------------------------------------
        if c.endswith("_weight"):
            x[c] = pd.to_numeric(s, errors="coerce").ffill().fillna(0.0)
            continue

        # ----------------------------------------
        # 2) UTC timestamp columns
        # ----------------------------------------
        if c.endswith("_utc") or c == "trained_at_utc":
            dt = pd.to_datetime(s, errors="coerce", utc=True)
            x[c] = dt.ffill()
            continue

        # ----------------------------------------
        # 3) ID-like columns
        # ----------------------------------------
        if c in id_like:
            x[c] = s.astype("string").ffill()
            continue

        # ----------------------------------------
        # 4) Numeric columns (model params, macro, gold)
        # ----------------------------------------
        if pd.api.types.is_numeric_dtype(s):
            x[c] = s.ffill()
            continue

        # ----------------------------------------
        # 5) Object/string columns (state_var etc.)
        # ----------------------------------------
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            x[c] = s.astype("string").ffill()
            continue

    return x


def join_daily(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left is None or left.empty:
        return left
    if right is None or right.empty:
        return left

    l = canonical_session_index(left)
    r = canonical_session_index(right)

    # now both are tz-naive midnight -> safe
    return l.join(r, how="left")
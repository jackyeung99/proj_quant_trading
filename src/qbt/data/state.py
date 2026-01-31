
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import hashlib
from typing import Optional
import pandas as pd


def get_last_available_date(
    storage,
    state_key: str,
    *,
    ts_field: str = "dataset_end",
) -> Optional[pd.Timestamp]:
    """
    Returns the last ingested timestamp from the state JSON, or None if missing.

    Assumes the timestamp is ISO-8601 and UTC or UTC-convertible.
    """
    if not storage.exists(state_key):
        return None

    state = storage.read_json(state_key)
    if not state or ts_field not in state:
        return None

    ts = pd.to_datetime(state[ts_field], utc=True, errors="coerce")
    if pd.isna(ts):
        return None

    return ts



def compute_fetch_window(
    *,
    last_date: Optional[datetime],
    lookback_days: int,
    mode: str,  # "append" | "backfill" | "full" etc
    start_override: Optional[str] = None,
    end_override: Optional[str] = None,
    freeze_now: Optional[datetime] = None,
):
    """
    Returns [start, end) (end exclusive) or whatever convention you use.
    - In tests, pass start_override/end_override (and optionally freeze_now).
    - In prod, call without overrides.

    Assumes UTC datetimes.
    """
    now = freeze_now or datetime.now(timezone.utc)



    # 1) explicit overrides win (best for tests)
    if start_override is not None or end_override is not None:

        start_raw = start_override or last_date or (now - timedelta(days=365))
        end_raw = end_override or now

        start = pd.to_datetime(start_raw, utc=True, errors="coerce")
        end = pd.to_datetime(end_raw, utc=True, errors="coerce")

        if pd.isna(start) or pd.isna(end):
            raise ValueError(f"Invalid datetime(s): start={start_raw}, end={end_raw}")

        if start >= end:
            raise ValueError(f"Invalid window: start {start} >= end {end}")

        return start, end
    
    # 2) production logic
    if mode == "full":
        # choose whatever makes sense for your dataset
        start = now - timedelta(days=365 * 5)
        end = now
        return start, end

    if mode == "append":
        if last_date is None:
            # first run; define a sensible default
            start = now - timedelta(days=365 * 5)
        else:
            start = last_date - timedelta(days=lookback_days)
        end = now
        if start >= end:
            raise ValueError(f"Invalid window: start {start} >= end {end}")
        return start, end

    raise ValueError(f"Unknown mode: {mode}")



def _fingerprint(obj: dict) -> str:
    """
    Stable hash for config dictionaries.
    """
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def update_state(
    storage,
    state_key: str,
    df: pd.DataFrame,
    *,
    pull_start,
    pull_end,
    meta: dict,
    timestamp_col: str = "timestamp",
) -> None:
    """
    Write per-source ingestion state.

    Definitions:
    - pull_*        : window attempted in THIS run
    - dataset_*     : coverage of the stored dataset AFTER merge
    - last_success  : operational timestamp

    Best-effort: failures here must not break ingestion.
    """
    try:
        # ----- dataset coverage from timestamp column -----
        if df is None or df.empty:
            dataset_start = None
            dataset_end = None
            rows_total = 0
            rows_in_pull_window = 0
        else:
            if timestamp_col not in df.columns:
                raise ValueError(f"Missing '{timestamp_col}' column in df for state update")

            ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce").dropna()
            dataset_start = ts.min() if not ts.empty else None
            dataset_end = ts.max() if not ts.empty else None
            rows_total = int(len(df))

            # Rows that fall inside the attempted pull window (if provided)
            if pull_start is None and pull_end is None:
                rows_in_pull_window = rows_total
            else:
                start = pd.to_datetime(pull_start, utc=True, errors="coerce") if pull_start is not None else None
                end = pd.to_datetime(pull_end, utc=True, errors="coerce") if pull_end is not None else None

                mask = pd.Series(True, index=df.index)
                ts_full = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")

                if start is not None:
                    mask &= ts_full >= start
                if end is not None:
                    mask &= ts_full <= end

                rows_in_pull_window = int(mask.fillna(False).sum())

        state = {
            # identity
            "store_key": state_key,

            # operational
            "last_success_utc": datetime.now(timezone.utc).isoformat(),
            "rows_total": rows_total,
            "rows_in_pull_window": rows_in_pull_window,

            # pull window (this run)
            "pull_start": pd.to_datetime(pull_start, utc=True).isoformat() if pull_start is not None else None,
            "pull_end": pd.to_datetime(pull_end, utc=True).isoformat() if pull_end is not None else None,

            # dataset coverage (all data on disk)
            "dataset_start": dataset_start.isoformat() if dataset_start is not None else None,
            "dataset_end": dataset_end.isoformat() if dataset_end is not None else None,

            # reproducibility
            "config_fingerprint": _fingerprint(meta),
        }


        storage.write_json(state, state_key)

    except Exception as e:
        # State must never break ingestion
        print(f"[WARN] Failed to update state for {state_key}: {e}")

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import hashlib
from typing import Optional
import pandas as pd


def get_last_available_date(
    storage,
    store_key: str
) -> Optional[pd.Timestamp]:
    """
    Returns the max timestamp present in stored data for this source,
    or None if no data exists yet.

    Assumes timestamps are UTC or convertible to UTC.
    """
    if not storage.exists(store_key):
        return None

    df = storage.read_parquet(store_key)
    if df is None or df.empty:
        return None


    ts = pd.to_datetime(df.index, utc=True, errors="coerce")
    ts = ts.dropna()

    if ts.empty:
        return None

    return ts.max()




def compute_fetch_window(
    *,
    last_date: Optional[datetime],
    lookback_days: int,
    mode: str,  # "append" | "backfill" | "full" etc
    start_override: Optional[datetime] = None,
    end_override: Optional[datetime] = None,
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
        start = start_override or (last_date or (now - timedelta(days=365)))
        end = end_override or now
        if start >= end:
            raise ValueError(f"Invalid window: start {start} >= end {end}")
        return start, end

    # 2) production logic
    if mode == "full":
        # choose whatever makes sense for your dataset
        start = now - timedelta(days=365 * 2)
        end = now
        return start, end

    if mode == "append":
        if last_date is None:
            # first run; define a sensible default
            start = now - timedelta(days=365 * 2)
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
    store_key: str,
    df: pd.DataFrame,
    *,
    pull_start,
    pull_end,
    meta: dict,
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
        # Normalize index timestamps
        ts = pd.to_datetime(df.index, utc=True, errors="coerce").dropna()

        dataset_start = ts.min() if not ts.empty else None
        dataset_end = ts.max() if not ts.empty else None

        state = {
            # identity
            "store_key": store_key,

            # operational
            "last_success_utc": datetime.now(timezone.utc).isoformat(),
            "rows_written": int(len(df.loc[pull_start: ])),

            # pull window (this run)
            "pull_start": pull_start.isoformat() if pull_start is not None else None,
            "pull_end": pull_end.isoformat() if pull_end is not None else None,

            # dataset coverage (all data on disk)
            "dataset_start": dataset_start.isoformat() if dataset_start is not None else None,
            "dataset_end": dataset_end.isoformat() if dataset_end is not None else None,

            # reproducibility
            "config_fingerprint": _fingerprint(meta),
        }

        state_key = f"{store_key}.state.json"
        storage.write_json(state, state_key)

    except Exception as e:
        # State must never break ingestion
        print(f"[WARN] Failed to update state for {store_key}: {e}")
import pandas as pd


def _ensure_tz(ts: pd.Series, tz: str) -> pd.Series:
    t = pd.to_datetime(ts, errors="coerce")
    # If naive, localize; if aware, convert.
    if getattr(t.dt, "tz", None) is None:
        return t.dt.tz_localize(tz)
    return t.dt.tz_convert(tz)


def _to_daily_index(ts: pd.Series) -> pd.DatetimeIndex:
    # Normalize to date (still tz-aware); use date boundary in exchange tz
    return pd.DatetimeIndex(ts.dt.normalize())
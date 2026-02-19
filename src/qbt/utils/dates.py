import pandas as pd
import exchange_calendars as xcals

def _ensure_tz(ts: pd.Series, tz: str) -> pd.Series:
    t = pd.to_datetime(ts, errors="coerce")
    # If naive, localize; if aware, convert.
    if getattr(t.dt, "tz", None) is None:
        return t.dt.tz_localize(tz)
    return t.dt.tz_convert(tz)


def _ensure_session_index(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if isinstance(x.index, pd.DatetimeIndex) and x.index.name == "session_date":
        x.index = pd.to_datetime(x.index).normalize()
        return x.sort_index()

    if "session_date" in x.columns:
        x["session_date"] = pd.to_datetime(x["session_date"], errors="coerce").dt.normalize()
        x = x.dropna(subset=["session_date"]).set_index("session_date").sort_index()
        return x

    raise ValueError("Expected 'session_date' as index or column.")

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

def _to_daily_index(ts: pd.Series) -> pd.DatetimeIndex:
    # Normalize to date (still tz-aware); use date boundary in exchange tz
    return pd.DatetimeIndex(ts.dt.normalize())

def _to_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")

def _session_date_from_ts(ts: pd.Timestamp, *, market_tz: str) -> pd.Timestamp:
    """
    Return session date as a pure date (YYYY-MM-DD),
    based on market timezone.
    """

    if ts.tzinfo is None:
        return ts.date()

    return ts.tz_convert(market_tz).normalize().tz_localize(None)


def _stamp_asof_utc(session_date: pd.Timestamp, *, market_tz: str, hour: float) -> pd.Timestamp:
    """
    session_date is naive midnight. Return tz-aware UTC timestamp at {hour}:00 market time.
    Supports fractional hours (e.g., 16.5).
    """
    h = int(hour)
    m = int(round((hour - h) * 60))
    return (session_date.tz_localize(market_tz) + pd.Timedelta(hours=h, minutes=m)).tz_convert("UTC")

def _parse_cutoff(cutoff_hour: float) -> pd.Timedelta:
    h = int(cutoff_hour)
    m = int(round((cutoff_hour - h) * 60))
    return pd.Timedelta(hours=h, minutes=m)


def roll_to_next_session_nyse(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    dates: timezone-aware or naive daily timestamps (we only use the date part)
    returns: dates rolled forward to the next NYSE session if needed
    """
    cal = xcals.get_calendar("XNYS")

    # Convert to naive dates for session lookup
    d = pd.DatetimeIndex(pd.to_datetime(dates).date)

    # cal.sessions is a DatetimeIndex of valid session opens (midnight UTC-ish is fine)
    sessions = cal.sessions  # all sessions in calendar range

    # For each date, find the first session >= that date
    # Using searchsorted on sessions (convert to naive date timestamps)
    sess = pd.DatetimeIndex(pd.to_datetime(sessions.date))
    pos = sess.searchsorted(d, side="left")
    pos = pos.clip(0, len(sess) - 1)

    return sess[pos]
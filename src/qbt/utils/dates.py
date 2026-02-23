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
    Return session date (tz-naive midnight) based on market timezone.
    """

    ts = pd.to_datetime(ts)

    # If tz-naive, assume it's UTC (your pipeline is UTC-first)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    return ts.tz_convert(market_tz).normalize().tz_localize(None)

def stamp_asof_utc_from_session_dates(
    session_dates: pd.DatetimeIndex,
    *,
    market_tz: str,
    cutoff_hour: float,
) -> pd.DatetimeIndex:
    """
    session_dates: tz-naive midnight labels.
    returns: tz-aware UTC timestamps at cutoff_hour local time.
    """
    sd = pd.DatetimeIndex(pd.to_datetime(session_dates, errors="coerce")).normalize()
    local = sd.tz_localize(market_tz) + pd.to_timedelta(cutoff_hour, unit="h")
    return local.tz_convert("UTC")


def _stamp_asof_utc(session_date: pd.Timestamp, *, market_tz: str, hour: float) -> pd.Timestamp:
    """
    session_date: intended to be a daily label (midnight).
      - If tz-aware, convert to market_tz then drop tz.
      - If tz-naive, treat as already a session label.
    Returns tz-aware UTC timestamp at {hour} local market time.
    """
    sd = pd.Timestamp(session_date)

    # If tz-aware, convert to market tz and drop tz (label)
    if sd.tzinfo is not None:
        sd = sd.tz_convert(market_tz).tz_localize(None)

    # normalize to midnight label
    sd = sd.normalize()

    h = int(hour)
    m = int(round((hour - h) * 60))

    local = sd.tz_localize(market_tz) + pd.Timedelta(hours=h, minutes=m)
    return local.tz_convert("UTC")

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
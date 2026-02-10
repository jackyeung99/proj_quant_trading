import pandas as pd
import exchange_calendars as xcals

def _ensure_tz(ts: pd.Series, tz: str) -> pd.Series:
    t = pd.to_datetime(ts, errors="coerce")
    # If naive, localize; if aware, convert.
    if getattr(t.dt, "tz", None) is None:
        return t.dt.tz_localize(tz)
    return t.dt.tz_convert(tz)


def _to_daily_index(ts: pd.Series) -> pd.DatetimeIndex:
    # Normalize to date (still tz-aware); use date boundary in exchange tz
    return pd.DatetimeIndex(ts.dt.normalize())


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
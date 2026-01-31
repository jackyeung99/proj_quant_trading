from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

import pandas as pd
import exchange_calendars as xcals

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


def _bucket_daily_with_cutoff(
    idx: pd.DatetimeIndex,
    *,
    cutoff_hour: float,
) -> pd.DatetimeIndex:
    cutoff = _parse_cutoff(cutoff_hour)
    offset = pd.Timedelta(days=1) - cutoff

    shifted = idx + offset
    day = shifted.normalize()  # daily labels

    # Roll to next NYSE session (handles weekends + holidays)
    day = roll_to_next_session_nyse(day)
    return day



@dataclass(frozen=True)
class IntraFeature:
    name: str
    requires: tuple[str, ...]
    func: Callable[[pd.DataFrame], float]   # computed on ONE bucket's intraday df


def compute_intra_features(
    bars: pd.DataFrame,
    features: Iterable[IntraFeature],
) -> pd.Series:
    """
    Compute many intraday features on a single bucket (one day/session).
    Returns Series indexed by feature.name.
    """
    out: dict[str, float] = {}
    cols = set(bars.columns)

    for feat in features:
        missing = [c for c in feat.requires if c not in cols]
        if missing:
            out[feat.name] = np.nan
            continue
        try:
            out[feat.name] = float(feat.func(bars))
        except Exception:
            out[feat.name] = np.nan

    return pd.Series(out)




def _parse_cutoff(cutoff_hour: float) -> pd.Timedelta:
    h = int(cutoff_hour)
    m = int(round((cutoff_hour - h) * 60))
    return pd.Timedelta(hours=h, minutes=m)





def aggregate_intraday_to_daily_features(
    df: pd.DataFrame,
    *,
    cutoff_hour: float = 15.0,
    tz: str | None = "America/New_York",
    features: Iterable[IntraFeature],
    keep_ohlc: bool = True,
    open_field: str = "open",
    close_field: str = "close",
) -> pd.DataFrame:
    """
    df: intraday bars for ONE asset, indexed by timestamp, columns like open/high/low/close/volume/etc.
    Returns: daily dataframe with feature columns (+ optional open/close).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")

    x = df.sort_index().copy()

    if tz is not None:
        if x.index.tz is None:
            x.index = x.index.tz_localize(tz)
        else:
            x.index = x.index.tz_convert(tz)

    bucket = _bucket_daily_with_cutoff(x.index, cutoff_hour=cutoff_hour)

    # compute features per bucket
    feats_df = x.groupby(bucket, sort=True).apply(lambda g: compute_intra_features(g, features))
    # groupby.apply returns a Series with MultiIndex; reshape to DataFrame
    feats_df = feats_df.unstack()

    if keep_ohlc:
        if open_field not in x.columns or close_field not in x.columns:
            raise ValueError(f"Missing {open_field}/{close_field} for keep_ohlc=True")
        ohlc = x.groupby(bucket).agg({open_field: "first", close_field: "last"})
        out = ohlc.join(feats_df, how="outer")
    else:
        out = feats_df

    return out.sort_index()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

import pandas as pd
import exchange_calendars as xcals

from qbt.features.apply import apply_intra_features

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



def _parse_cutoff(cutoff_hour: float) -> pd.Timedelta:
    h = int(cutoff_hour)
    m = int(round((cutoff_hour - h) * 60))
    return pd.Timedelta(hours=h, minutes=m)



def aggregate_intraday_to_daily_features(
    df: pd.DataFrame,
    *,
    cutoff_hour: float = 16.0,
    tz: str | None = "America/New_York",
    features: Iterable[Mapping[str, Any]],
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

    # --- build features: each spec becomes one column ---
    feat_series: list[pd.Series] = []
    cols = set(x.columns)

    for feat in features:
        name = str(feat.get("name"))
        requires = tuple(feat.get("requires", ()))  # type: ignore
        func = feat.get("func")

        if not name or func is None:
            continue

        # If required cols missing, return NaN for all buckets
        if any(c not in cols for c in requires):
            s = pd.Series(index=pd.Index(bucket.unique()).sort_values(), dtype="float64", name=name)
            s.loc[:] = np.nan
            feat_series.append(s)
            continue

        # Reduce intraday -> scalar per bucket
        def _safe_reduce(g: pd.DataFrame) -> float:
            try:
                return float(func(g))
            except Exception:
                return np.nan

        s = x.groupby(bucket, sort=True).apply(_safe_reduce)
        s.name = name
        feat_series.append(s)

    feats_df = pd.concat(feat_series, axis=1) if feat_series else pd.DataFrame(index=pd.Index(bucket.unique()).sort_values())

    # --- add OHLC if requested ---
    if keep_ohlc:
        if open_field not in x.columns or close_field not in x.columns:
            raise ValueError(f"Missing {open_field}/{close_field} for keep_ohlc=True")
        ohlc = x.groupby(bucket, sort=True).agg(
                        {
                            open_field: "first",
                            "high": "max",
                            "low": "min",
                            close_field: "last",
                            "volume": "sum",
                        }
                    )
        out = ohlc.join(feats_df, how="outer")
    else:
        out = feats_df

    return out.sort_index()
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

import pandas as pd

from qbt.features.apply import apply_intra_features
from qbt.utils.dates import roll_to_next_session_nyse, _parse_cutoff


def resample_ohlcv(
    df: pd.DataFrame,
    *,
    rv_freq: str = "10min",
    tz: str | None = "America/New_York",
    session_start: str = "09:30",
    session_end: str = "16:00",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    x = df.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        raise ValueError("df must be indexed by a DatetimeIndex.")
    x = x.sort_index()

    tz = tz or "America/New_York"  

    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC").tz_convert(tz)
    else:
        x.index = x.index.tz_convert(tz)

    if session_start and session_end:
        x = x.between_time(session_start, session_end, inclusive="both")

    needed = [open_col, high_col, low_col, close_col]
    missing = [c for c in needed if c not in x.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    agg_map = {open_col: "first", high_col: "max", low_col: "min", close_col: "last"}
    if volume_col in x.columns:
        agg_map[volume_col] = "sum"

    out = x.resample(rv_freq).agg(agg_map)
    out = out.dropna(subset=[open_col, high_col, low_col, close_col], how="all")
    return out




def bucket_daily_with_cutoff_with_roll(
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


def bucket_daily_with_cutoff_no_roll(
    idx: pd.DatetimeIndex,
    *,
    cutoff_hour: float,
) -> tuple[pd.DatetimeIndex, pd.Series]:
    """
    Return (labels, mask) where:
      - labels: calendar-day label for each timestamp (idx.normalize())
      - mask: boolean Series indicating rows to KEEP (time-of-day <= cutoff)
    """
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("idx must be a DatetimeIndex")

    cutoff = _parse_cutoff(cutoff_hour)

    # time since midnight (TimedeltaIndex)
    tod = idx - idx.normalize()

    # keep rows at or before cutoff
    mask = tod <= cutoff

    labels = idx.normalize()
    # keep index/labels aligned and preserve tz if present
    if idx.tz is not None and labels.tz is None:
        labels = labels.tz_localize(idx.tz)

    return labels, pd.Series(mask, index=idx)



def aggregate_intraday_to_daily_features(
    df: pd.DataFrame,
    *,
    intra_freq = '5 Min',
    cutoff_hour: float = 16.0,
    tz: str | None = "America/New_York",
    features: Iterable[Mapping[str, Any]],
    open_field: str = "open",
    close_field: str = "close",
) -> pd.DataFrame:
    """
    df: intraday bars for ONE asset, indexed by timestamp, columns like open/high/low/close/volume/etc.
    Returns: daily dataframe with feature columns (+ optional open/close).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")

    df = df.sort_index().copy()

    # resample to intraday freq    
    intra = resample_ohlcv(df, rv_freq=intra_freq, tz=tz)



    # ======================             

    '''
    TO DO: handle different ways to group intraday rows 

    ex.
    
    1. take days before cutoff for day t and rows after the cutoff go to day t+1 
    2. support different intraday freq use daily for intra computation if the end frequency is weekly 

    '''
    # handle cutoff 
    labels, mask = bucket_daily_with_cutoff_no_roll(intra.index, cutoff_hour=cutoff_hour)

    # apply mask to intra and labels
    intra_cut = intra.loc[mask.values]
    if intra_cut.empty:
        # no data after masking — return empty DataFrame with expected index type
        empty_index = pd.Index([], dtype="datetime64[ns]")
        return pd.DataFrame(index=empty_index)

    labels_cut = labels[mask.values]

    # ======================




    # build features
    feats_df = apply_intra_features(intra_cut, labels_cut, features)

    # --- add OHLC ---
    agg_map = {
        open_field: "first",
        "high": "max",
        "low": "min",
        close_field: "last",
    }
    if "volume" in intra_cut.columns:
        agg_map["volume"] = "sum"

    ohlc = intra_cut.groupby(labels_cut, sort=True).agg(agg_map)
    out = ohlc.join(feats_df, how="outer")
    

    return out.sort_index()





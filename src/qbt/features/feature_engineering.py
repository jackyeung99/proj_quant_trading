from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from qbt.core.types import ModelInputs, RunSpec
from qbt.features.transforms import *
from qbt.features.feature_engineering import *



def aggregate_intra_bars(
    prices: pd.DataFrame,
    *,
    freq: str = "1B",
    cutoff_hour: float = 15.0,          # e.g. 15.0 -> day ends at 15:00
    return_kind: str = "log",
    tz: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate intraday bars to a lower frequency with a custom cutoff.

    Parameters
    ----------
    prices : DataFrame
        Wide DataFrame indexed by tz-aware timestamps (bar CLOSE), columns=assets.
        Must contain price levels (close). We'll infer open/close per period from prices.
    freq : str
        Resample frequency, usually "1D".
    cutoff_hour : float
        End-of-period cutoff hour in local time. Example: 15.0 means the "day"
        runs from (prev day 15:15 .. current day 15:00) if 15m bars are close-stamped.
    return_kind : str
        "log" or "simple" intrabar returns.
    tz : str | None
        If given, convert/localize prices index to tz before resampling.

    Returns
    -------
    DataFrame with MultiIndex columns (asset, metric):
      open, close, ret, rvar, rvol, n_intra
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must be indexed by DatetimeIndex")
    if prices.index.tz is None and tz is None:
        raise ValueError("prices index must be tz-aware, or pass tz=... to localize.")
    if tz is not None:
        prices = prices.tz_localize(tz) if prices.index.tz is None else prices.tz_convert(tz)

    prices = prices.sort_index()

    # intrabar returns (aligned to bar close times)
    r = compute_returns(prices, kind=return_kind)

    # realized variance components
    r_sq = r.pow(2)

    # resample offset:
    # pandas offset shifts the bin *boundaries*. To make the "day end" at cutoff_hour,
    # shift boundaries back by cutoff_hour (negative).
    hours = int(cutoff_hour)
    minutes = int(round((cutoff_hour - hours) * 60))
    boundary_shift = pd.Timedelta(hours=hours, minutes=minutes)

    # build per-asset aggregates
    frames = []
    for a in prices.columns:
        df = pd.DataFrame(
            {
                "price": prices[a],
                "r": r[a],
                "r_sq": r_sq[a],
            }
        )       


        agg = (
            df.resample(freq, offset=boundary_shift)
              .agg(
                  open=("price", "first"),
                  close=("price", "last"),
                  ret=("r", "sum"),
                  rvar=("r_sq", "sum"),
                  n_intra=("r", "count"),
              )
        )
        agg["rvol"] = np.sqrt(agg["rvar"])

        # put into (asset, metric) columns
        agg.columns = pd.MultiIndex.from_product([[a], agg.columns])
        frames.append(agg)

    out = pd.concat(frames, axis=1).sort_index()
    return out





def build_feature_frame(
    prices: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    rv_windows: Sequence[int] = (),
    mom_windows: Sequence[int] = (),
    vol_windows: Sequence[int] = (),
    col_style: str = "flat",  # "flat" or "multiindex"
) -> pd.DataFrame:
    """
    Build a generic multi-asset feature frame.

    Index: time
    Columns:
      - flat: ASSET__feature
      - multiindex: (asset, feature)
    """
    feats = []

    # --- Volatility / risk ---
    for w in rv_windows:
        rv = realized_vol(ret, w)
        rv.columns = pd.MultiIndex.from_product([rv.columns, [f"rv_{w}"]])
        feats.append(rv)

    # --- Direction / momentum ---
    for w in mom_windows:
        mom = momentum(prices, w)
        mom.columns = pd.MultiIndex.from_product([mom.columns, [f"mom_{w}"]])
        feats.append(mom)

    # --- Distribution (optional) ---
    for w in vol_windows:
        std = ret.rolling(w).std()
        std.columns = pd.MultiIndex.from_product([std.columns, [f"std_{w}"]])
        feats.append(std)

    if not feats:
        raise ValueError("No features requested")

    X = pd.concat(feats, axis=1).sort_index(axis=1)

    if col_style == "flat":
        X.columns = [f"{a}__{f}" for a, f in X.columns]

    return X

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
    wide: pd.DataFrame,
    *,
    freq: str = "1B",
    cutoff_hour: float = 15.0,
    return_kind: str = "log",
    tz: str | None = None,
    open_field: str = "open",
    close_field: str = "close",
) -> pd.DataFrame:
    """
    Aggregate intraday bars (wide) to a lower frequency with a custom cutoff.

    Accepts:
      1) close-only wide:
         index = tz-aware timestamps, columns = assets, values = close
      2) OHLC wide (MultiIndex columns):
         index = tz-aware timestamps, columns = (asset, field), field in {"open","close",...}

    Returns DataFrame with MultiIndex columns (asset, metric):
      open, close, ret, rvar, rvol, n_intra
    """
    if not isinstance(wide.index, pd.DatetimeIndex):
        raise ValueError("wide must be indexed by DatetimeIndex")

    if wide.index.tz is None and tz is None:
        raise ValueError("wide index must be tz-aware, or pass tz=... to localize.")

    # normalize timezone
    if tz is not None:
        wide = wide.tz_localize(tz) if wide.index.tz is None else wide.tz_convert(tz)

    wide = wide.sort_index()

    # ----------------------------
    # Extract per-asset intraday open/close series
    # ----------------------------
    if isinstance(wide.columns, pd.MultiIndex):
        # Expect columns like (asset, field)
        levels = wide.columns.names
        # We assume level 0 = asset, level 1 = field (common in your code)
        # If your ordering differs, you can reorder outside before calling.
        fields = set(wide.columns.get_level_values(1))

        has_open = open_field in fields
        has_close = close_field in fields

        if not has_close:
            raise ValueError(f"MultiIndex wide must contain field '{close_field}'.")

        close_w = wide.xs(close_field, level=1, axis=1)

        if has_open:
            open_w = wide.xs(open_field, level=1, axis=1)
        else:
            open_w = None

    else:
        # close-only
        close_w = wide
        open_w = None

    close_w = close_w.sort_index()

    # If we have intraday open, align columns to close columns
    if open_w is not None:
        open_w = open_w.reindex(columns=close_w.columns).sort_index()

    # ----------------------------
    # Intrabar returns for RV (use CLOSE series)
    # ----------------------------
    r = compute_returns(close_w, kind=return_kind)  # aligned to bar timestamps
    r_sq = r.pow(2)

    # boundary shift so bins "end" at cutoff_hour in local tz
    hours = int(cutoff_hour)
    minutes = int(round((cutoff_hour - hours) * 60))
    boundary_shift = pd.Timedelta(hours=hours, minutes=minutes)

    frames: list[pd.DataFrame] = []

    for a in close_w.columns:
        price_c = close_w[a]

        # open/close per period:
        # - close: last close within the bin
        # - open: if intraday open is available, use first open within the bin
        #         else fall back to first close within the bin
        if open_w is not None and a in open_w.columns:
            price_o = open_w[a]
            src_open = "open"
        else:
            price_o = price_c
            src_open = "close"  # inferred open from first close

        df = pd.DataFrame(
            {
                "open_src": price_o,
                "close_src": price_c,
                "r": r[a],
                "r_sq": r_sq[a],
            }
        )

        agg = (
            df.resample(freq, offset=boundary_shift)
              .agg(
                  open=("open_src", "first"),
                  close=("close_src", "last"),
                  ret=("r", "sum"),
                  rvar=("r_sq", "sum"),
                  n_intra=("r", "count"),
              )
        )

        agg["rvol"] = np.sqrt(agg["rvar"])
        agg["open_is_inferred"] = 0 if src_open == "open" else 1  # handy debug flag

        # (asset, metric) columns
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

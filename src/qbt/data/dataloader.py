from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from qbt.core.types import ModelInputs, RunSpec
from qbt.features.transforms import *
from qbt.features.feature_engineering import *

# ----------------------------
# IO: load long OHLCV, then pivot to wide
# ----------------------------

def load_multi_asset_flat_long(
    root: str | Path,
    bar: str,                      # "15min", "1d"
    assets: Sequence[str],
    file_format: str = "parquet",
    timestamp_col: str = "timestamp",
    fields: Sequence[str] = ("close",),
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Returns LONG data:
      columns: [timestamp, asset, <fields...>]
    """
    root = Path(root)
    frames: list[pd.DataFrame] = []

    ext = "parquet" if file_format == "parquet" else "csv"

    for a in assets:
        path = root/f"freq={bar}"/f"ticker={a}/bars.{ext}"

        print(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing data for asset={a}: {path}")

        if ext == "parquet":
            # only read needed fields + timestamp
            cols = [timestamp_col, *fields]
            df = pd.read_parquet(path, columns=cols)
        else:
            df = pd.read_csv(path)

        if timestamp_col in df.columns:
            ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        else:
            ts = pd.to_datetime(df.index, errors="coerce")

        out = df.copy()
        out[timestamp_col] = ts
        out["asset"] = a
        out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

        # tz normalize
        t = out[timestamp_col]
        if getattr(t.dt, "tz", None) is None:
            out[timestamp_col] = t.dt.tz_localize(tz)
        else:
            out[timestamp_col] = t.dt.tz_convert(tz)

        keep = [timestamp_col, "asset", *[c for c in fields if c in out.columns]]
        frames.append(out[keep])

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["asset", timestamp_col])
    return raw


def long_to_wide(
    raw_long: pd.DataFrame,
    timestamp_col: str = "timestamp",
    asset_col: str = "asset",
    value_cols: Sequence[str] = ("close",),
) -> dict[str, pd.DataFrame]:
    """
    Returns dict[field] = wide DataFrame(index=time, columns=assets)
    """
    out: dict[str, pd.DataFrame] = {}
    for c in value_cols:
        if c not in raw_long.columns:
            raise ValueError(f"Missing required column '{c}' in raw data.")
        wide = (
            raw_long[[timestamp_col, asset_col, c]]
            .pivot(index=timestamp_col, columns=asset_col, values=c)
            .sort_index()
        )
        out[c] = wide
    return out




# ----------------------------
# Adapter
# ----------------------------

class DataAdapter:
    def load(self, spec: RunSpec) -> pd.DataFrame: ...
    def prepare(self, raw: pd.DataFrame, spec: RunSpec, required_cols: list[str]) -> ModelInputs: ...


@dataclass
class DefaultDataAdapter(DataAdapter):
    """
    - load() returns LONG raw OHLCV
    - prepare() returns ModelInputs with clean separation
    """

    def load(self, spec: RunSpec) -> pd.DataFrame:
        d = spec.data  # assumes spec.data is dict-like (or a dataclass)

        return load_multi_asset_flat_long(
            root=d["root"],
            bar=str(d.get('interval', '1d')),          # or spec.time.bar
            assets=spec.assets,
            file_format=d.get("file_format", "parquet"),
            timestamp_col=d.get("columns", {}).get("timestamp", "timestamp"),
            fields=tuple(d.get("columns", {}).get("fields", ["close"])),
            tz=d.get("tz", "America/New_York"),
        )

    def prepare(self, raw: pd.DataFrame, spec: RunSpec, required_cols: list[str]) -> ModelInputs:
        # 1) pivot to wide
        ts_col = spec.data.get("columns", {}).get("timestamp", "timestamp")
        fields = tuple(spec.data.get("columns", {}).get("fields", ["open", "close"]))
        wide = long_to_wide(raw, timestamp_col=ts_col, value_cols=fields)


        # TIMING STRATEGY Specific abstract later

        open_15m = wide.get("open")
        close_15m = wide.get("close")
        if open_15m is None or close_15m is None:
            raise ValueError("Expected 'open' and 'close' in data.columns.fields.")

        open_15m = open_15m.sort_index()
        close_15m = close_15m.sort_index()

        # 2) intraday close-to-close returns (15m) for RV computations
        # NOTE: this is NOT what you use for 'trade next open' execution; it's for features.
        p = spec.features  # dict-like
        tz = p.get("tz", "America/New_York")
        return_kind = p.get("return_kind", "log")

        # 3) aggregate to daily using close series (your current approach)
        daily = aggregate_intra_bars(
            prices=close_15m,
            freq="1B",
            cutoff_hour=15.0,
            return_kind=return_kind,
            tz=tz,
        ).dropna(how="any")

        # 4) daily open/close (wide: cols=assets)
        daily_open  = daily.xs("open",  level=1, axis=1)
        daily_close = daily.xs("close", level=1, axis=1)

        # 5) daily close-to-close return (standard diff, aligned to day t close)
        ret_cc = compute_returns(daily_close, kind=return_kind)

        # 6) daily open-to-open return for "trade next open" execution
        # ret_oo[t] corresponds to open[t] -> open[t+1], aligned to t
        ret_oo = compute_returns(daily_open, kind=return_kind)

        # 7) features (example: daily rvol from aggregator, flattened)
        X_parts = []
        if ("XLE", "rvol") in daily.columns or any(m == "rvol" for _, m in daily.columns):
            rvol = daily.xs("rvol", level=1, axis=1)  # wide cols=assets
            rvol.columns = [f"{c}_rvol" for c in rvol.columns]
            X_parts.append(rvol)

        X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=daily.index)

        # 8) required feature columns check
        if required_cols:
            missing = [c for c in required_cols if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")

        # 9) align indices (important!)
        # Choose a single daily index where everything exists.
        # ret_oo is shorter by 1 row (no last open->open), so intersect.
        idx = daily.index
        idx = idx.intersection(ret_cc.index)
        idx = idx.intersection(ret_oo.index)  # drops last day automatically

        daily_open = daily_open.loc[idx]
        daily_close = daily_close.loc[idx]
        ret_cc = ret_cc.loc[idx]
        ret_oo = ret_oo.loc[idx]
        X = X.reindex(idx)

        # intraday returns: keep full intraday index, but you may also want to trim
        # to cover only the same date range as idx (optional)
        # ret_intra_cc = ret_intra_cc.loc[ret_intra_cc.index.min():ret_intra_cc.index.max()]

        return ModelInputs(
            ret=ret_oo,
            features=X,
        )
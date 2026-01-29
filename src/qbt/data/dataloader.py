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


        keep = [timestamp_col, "asset", *[c for c in fields if c in out.columns]]
        frames.append(out[keep])

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["asset", timestamp_col])
    return raw


def long_to_wide(
    raw_long: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    asset_col: str = "asset",
    value_cols: Sequence[str] = ("close",),
) -> pd.DataFrame:
    """
    Convert long OHLCV-style data into a wide DataFrame with MultiIndex columns.

    Parameters
    ----------
    raw_long : DataFrame
        Columns: [timestamp_col, asset_col, <value_cols...>]
    value_cols : Sequence[str]
        Fields to pivot (e.g. ["open", "close", "volume"])

    Returns
    -------
    wide : DataFrame
        index   = DatetimeIndex (tz-aware)
        columns = MultiIndex (asset, field)
    """
    missing = {timestamp_col, asset_col, *value_cols} - set(raw_long.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw_long: {missing}")

    frames: list[pd.DataFrame] = []

    for field in value_cols:
        w = (
            raw_long[[timestamp_col, asset_col, field]]
            .pivot(index=timestamp_col, columns=asset_col, values=field)
            .sort_index()
        )

        # attach field level
        w.columns = pd.MultiIndex.from_product(
            [w.columns, [field]],
            names=[asset_col, "field"],
        )
        frames.append(w)

    # concatenate along columns
    wide = pd.concat(frames, axis=1).sort_index()

    # ensure consistent column ordering: (asset, field)
    wide = wide.sort_index(axis=1, level=[0, 1])

    return wide




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
        )

    def prepare(self, raw: pd.DataFrame, spec: RunSpec, required_cols: list[str]) -> ModelInputs:
        # 1) pivot to wide (expects raw timestamps already tz-aware UTC)
        ts_col = spec.data.get("columns", {}).get("timestamp", "timestamp")
        fields = tuple(spec.data.get("columns", {}).get("fields", ["open", "close"]))
        wide = long_to_wide(raw, timestamp_col=ts_col, value_cols=fields)

        # --- feature config ---
        p = spec.features  # dict-like
        market_tz = p.get("tz", "America/New_York")
        return_kind = p.get("return_kind", "log")
        cutoff_hour = float(p.get("cutoff_hour", 14.0))  # optional override

     
        # 4) aggregate to daily in market tz (returns a DataFrame with MultiIndex columns: (asset, metric))
        daily_mkt = aggregate_intra_bars(
            wide=wide,
            freq="1B",
            cutoff_hour=cutoff_hour,
            return_kind=return_kind,
            tz=market_tz,
        ).dropna(how="any")


        # ensure expected metrics exist
        if not isinstance(daily_mkt.columns, pd.MultiIndex):
            raise ValueError("aggregate_intra_bars() must return MultiIndex columns like (asset, metric).")
        metrics = set(daily_mkt.columns.get_level_values(1))
        if "open" not in metrics or "close" not in metrics:
            raise ValueError("aggregate_intra_bars() output must include metrics 'open' and 'close'.")

        # 5) daily open/close (in market tz), then convert index back to UTC
        daily_open_mkt = daily_mkt.xs("open", level=1, axis=1)
        daily_close_mkt = daily_mkt.xs("close", level=1, axis=1)

        daily_open = daily_open_mkt.tz_convert("UTC")
        daily_close = daily_close_mkt.tz_convert("UTC")

        # 6) returns
        ret_cc = compute_returns(daily_close, kind=return_kind)
        ret_oo = compute_returns(daily_open, kind=return_kind)

        # 7) features: rvol from daily_mkt (market tz), then convert to UTC index
        X_parts: list[pd.DataFrame] = []
        if "rvol" in metrics:
            rvol_mkt = daily_mkt.xs("rvol", level=1, axis=1)
            rvol = rvol_mkt.tz_convert("UTC")
            rvol.columns = [f"{c}_rvol" for c in rvol.columns]
            X_parts.append(rvol)

        X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=daily_close.index)

        # 8) required feature columns check
        if required_cols:
            missing = [c for c in required_cols if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")

        # 9) align indices (drop last day because of shift(-1) -> NaN)
        idx = daily_close.index
        idx = idx.intersection(ret_cc.dropna(how="all").index)
        idx = idx.intersection(ret_oo.dropna(how="all").index)
        idx = idx.intersection(X.index)

        daily_open = daily_open.loc[idx]
        daily_close = daily_close.loc[idx]
        ret_cc = ret_cc.loc[idx]
        ret_oo = ret_oo.loc[idx]
        X = X.reindex(idx)

        return ModelInputs(
            ret=ret_oo,   # execution return stream (open->open aligned to decision day t)
            features=X,   # features indexed on same day t (UTC)
        )
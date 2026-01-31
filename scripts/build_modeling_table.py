from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Iterable

import numpy as np
import pandas as pd

from qbt.features.feature_engineering import aggregate_intra_bars
from qbt.features.transforms import compute_returns, estimate_idiosyncratic_multiindex


# -----------------------------
# Minimal "config" for now
# (move to config.py later)
# -----------------------------
@dataclass(frozen=True)
class BuildSpec:
    root: Path                     # data root containing freq=15min/ticker=.../bars.parquet
    assets: tuple[str, ...]
    bar_freq: str = "15min"        # folder name in freq=<bar_freq>
    out_dir: Path = Path("data/gold/model_table/universe=DEFAULT/freq=1D")
    file_format: str = "parquet"   # parquet or csv
    timestamp_col: str = "timestamp"
    fields: tuple[str, ...] = ("open", "close")  # must include close; open required if you want ret_oo

    # aggregation / market conventions
    market_tz: str = "America/New_York"
    cutoff_hour: float = 14.0
    agg_freq: str = "1B"           # business day buckets
    return_kind: str = "log"       # "log" or "simple"

    # features to include in modeling table
    include_rvol: bool = True      # uses "rvol" output from aggregate_intra_bars if present


# -----------------------------
# IO helpers
# -----------------------------
def load_multi_asset_flat_long(
    root: str | Path,
    bar: str,
    assets: Sequence[str],
    *,
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
        path = root / f"freq={bar}" / f"ticker={a}" / f"bars.{ext}"
        if not path.exists():
            raise FileNotFoundError(f"Missing data for asset={a}: {path}")

        cols = [timestamp_col, *fields]
        if ext == "parquet":
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

    # de-dupe on (asset, timestamp) keeping last (common cleanup step)
    raw = raw.drop_duplicates(subset=["asset", timestamp_col], keep="last")
    return raw


def long_to_wide(
    raw_long: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    asset_col: str = "asset",
    value_cols: Sequence[str] = ("close",),
) -> pd.DataFrame:
    """
    Convert long OHLC-style data into a wide DataFrame with MultiIndex columns.
    index   = DatetimeIndex
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
        w.columns = pd.MultiIndex.from_product(
            [w.columns, [field]],
            names=[asset_col, "field"],
        )
        frames.append(w)

    wide = pd.concat(frames, axis=1).sort_index()
    wide = wide.sort_index(axis=1, level=[0, 1])
    return wide


# -----------------------------
# Build modeling table (daily, multi-asset)
# -----------------------------
def _spec_hash(spec: BuildSpec) -> str:
    d = asdict(spec)
    # normalize paths to strings for stable hashing
    d["root"] = str(d["root"])
    d["out_dir"] = str(d["out_dir"])
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]




def build_daily_from_15m(
    raw_15m_long: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    asset_col: str = "asset",
    fields: tuple[str, ...] = ("open", "close"),
    market_tz: str = "America/New_York",
    cutoff_hour: float = 14.0,
    agg_freq: str = "1B",
    return_kind: str = "log",
) -> dict[str, pd.DataFrame]:
    """
    Reads 15m LONG data and produces daily WIDE series per asset:

    Returns dict with keys:
      open, close, ret_cc, ret_oo, intra_rvol, intra_rvar

    Each value is a DataFrame:
      index   = daily timestamps (UTC midnight normalized)
      columns = assets
    """
    # --- pivot long -> wide intraday for open/close ---
    missing = {timestamp_col, asset_col, *fields} - set(raw_15m_long.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw_15m_long: {missing}")

    ts = pd.to_datetime(raw_15m_long[timestamp_col], errors="coerce")
    df = raw_15m_long.copy()
    df[timestamp_col] = ts
    df = df.dropna(subset=[timestamp_col, asset_col]).sort_values([asset_col, timestamp_col])
    df = df.drop_duplicates(subset=[asset_col, timestamp_col], keep="last")

    # Build wide intraday MultiIndex columns: (asset, field)
    wide_parts = []
    for f in fields:
        w = df.pivot(index=timestamp_col, columns=asset_col, values=f).sort_index()
        w.columns = pd.MultiIndex.from_product([w.columns, [f]], names=[asset_col, "field"])
        wide_parts.append(w)
    wide = pd.concat(wide_parts, axis=1).sort_index().sort_index(axis=1)


    # --- aggregate to daily in market tz ---
    daily_mkt = aggregate_intra_bars(
        wide=wide,
        freq=agg_freq,
        cutoff_hour=float(cutoff_hour),
        return_kind=return_kind,
        tz=market_tz,
    ).sort_index().dropna(how='any')

    if not isinstance(daily_mkt.columns, pd.MultiIndex):
        raise ValueError("aggregate_intra_bars must return MultiIndex columns like (asset, metric).")

    metrics = set(daily_mkt.columns.get_level_values(1))

    # Minimal required outputs from aggregator
    for m in ("open", "close"):
        if m not in metrics:
            raise ValueError(f"aggregate_intra_bars output must include metric '{m}'.")

    # --- pull daily open/close, convert index to UTC midnight ---
    open_utc = daily_mkt.xs("open", level=1, axis=1).tz_convert("UTC")
    close_utc = daily_mkt.xs("close", level=1, axis=1).tz_convert("UTC")
    open_utc.index = open_utc.index.normalize()
    close_utc.index = close_utc.index.normalize()

    # --- compute returns ---
    ret_cc = compute_returns(close_utc, kind=return_kind)
    ret_oo = compute_returns(open_utc, kind=return_kind)

    # --- intraday realized variance/volatility ---
    # If your aggregator already provides rvol/rvar, use it.
    # Otherwise: derive rvar from 15m close-to-close log returns within each market day.
    intra_rvar = None
    intra_rvol = None

    if "rvar" in metrics:
        intra_rvar = daily_mkt.xs("rvar", level=1, axis=1).tz_convert("UTC")
        intra_rvar.index = intra_rvar.index.normalize()
    if "rvol" in metrics:
        intra_rvol = daily_mkt.xs("rvol", level=1, axis=1).tz_convert("UTC")
        intra_rvol.index = intra_rvol.index.normalize()

    # --- align indices across outputs ---
    idx = open_utc.index
    idx = idx.intersection(close_utc.index)
    idx = idx.intersection(ret_cc.index).intersection(ret_oo.index)
    idx = idx.intersection(intra_rvar.index).intersection(intra_rvol.index)

    open_utc = open_utc.loc[idx]
    close_utc = close_utc.loc[idx]
    ret_cc = ret_cc.loc[idx]
    ret_oo = ret_oo.loc[idx]
    intra_rvar = intra_rvar.loc[idx]
    intra_rvol = intra_rvol.loc[idx]

    return {
        "open": open_utc,
        "close": close_utc,
        "ret_cc": ret_cc,
        "ret_oo": ret_oo,
        "rvar": intra_rvar,
        "rvol": intra_rvol,
    }


def pack_daily_panels_to_one_table(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine outputs of build_daily_from_15m into one wide DataFrame
    with MultiIndex columns: (asset, field).
    """
    parts = []
    for field, df in panels.items():
        tmp = df.copy()
        tmp.columns = pd.MultiIndex.from_product([tmp.columns, [field]], names=["asset", "field"])
        parts.append(tmp)
    out = pd.concat(parts, axis=1).sort_index()
    out = out.sort_index(axis=1, level=[0, 1])
    return out


# -----------------------------
# Persist + manifest
# -----------------------------

def write_gold_model_table(df: pd.DataFrame, spec: BuildSpec, *, spec_hash: str) -> None:
    """
    Writes a WIDE modeling table:
      - df.index: DatetimeIndex (daily, UTC-normalized recommended)
      - df.columns: flattened strings like '<ASSET>_<field>'

    Outputs:
      - model.parquet  (index preserved)
      - _manifest.json
    """
    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected df.index to be a DatetimeIndex for wide modeling table.")
    if df.index.tz is not None:
        # ok, but store UTC in manifest. (Your pipeline normalizes already.)
        idx_utc = df.index.tz_convert("UTC")
    else:
        # assume it's already UTC-normalized naive
        idx_utc = df.index

    if df.index.has_duplicates:
        raise ValueError("df.index has duplicates; cannot write tidy daily modeling table.")

    # Write parquet WITH index (date)
    out_path = out_dir / "model.parquet"
    df.to_parquet(out_path, index=True)

    # infer assets
    assets = sorted(df.columns.get_level_values(0).unique().tolist())

    # manifest schema summary (columns + dtype)
    cols_meta = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]

    manifest = {
        "dataset": "gold_model_table",
        "format": "wide_flat",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_hash": spec_hash,
        "spec": {
            **asdict(spec),
            "root": str(spec.root),
            "out_dir": str(spec.out_dir),
            "assets": list(assets),
        },
        "shape": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "assets": int(len(assets)),
            "start_date_utc": idx_utc.min().isoformat(),
            "end_date_utc": idx_utc.max().isoformat(),
        },
        "index": {
            "name": df.index.name or "date",
            "dtype": str(df.index.dtype),
            "tz": "UTC" if df.index.tz is not None else None,
        },
        "columns": cols_meta,
        "asof_convention": (
            "Row at date=t contains fields/features computed using data available up to t. "
            "Execution shifting should happen in portfolio simulation (weights shifted to t+1)."
        ),
        "paths": {
            "model_parquet": str(out_path),
            "manifest_json": str(out_dir / "_manifest.json"),
        },
    }

    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    spec = BuildSpec(
        root=Path("data/bronze"),  # <-- change to your base (bronze or silver)
        assets=("XLE",  ),  # <-- set your assets
        bar_freq="15Min",
        out_dir=Path("data/gold/"),
        file_format="parquet",
        timestamp_col="timestamp",
        fields=("open", "close"),
        market_tz="America/New_York",
        cutoff_hour=14.0,
        agg_freq="1B",
        return_kind="log",
        include_rvol=True,
    )

    h = _spec_hash(spec)

    raw = load_multi_asset_flat_long(
        root=spec.root,
        bar=spec.bar_freq,
        assets=spec.assets,
        file_format=spec.file_format,
        timestamp_col=spec.timestamp_col,
        fields=spec.fields,
    )

    # panels = build_daily_from_15m(raw_15m_long, market_tz="America/New_York", cutoff_hour=14.0)

    panels = build_daily_from_15m(raw, market_tz = spec.market_tz, cutoff_hour=spec.cutoff_hour)
    
    model = pack_daily_panels_to_one_table(panels)

    print(model)
    write_gold_model_table(model, spec, spec_hash=h)

    print(f"Wrote: {spec.out_dir / 'model.parquet'}")
    print(f"Wrote: {spec.out_dir / '_manifest.json'}")
    # print(f"spec_hash={h}  rows={len(model)}  assets={model['asset'].nunique()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Sequence

import pandas as pd

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths
from qbt.storage.feature_store import write_gold_long_with_manifest

from qbt.features.aggregations import aggregate_intraday_to_daily_features
from qbt.features.apply import apply_transforms
from qbt.features.transforms import DAILY_TRANSFORMS
from qbt.features.intraday_transforms import INTRA_FEATURE_FUNCS
from qbt.data.loaders import load_multi_asset_flat_long






def build_intra_feature_specs(gold_cfg: dict) -> list[dict]:
    feats_cfg = gold_cfg.get("intra_features", []) or []
    out: list[dict] = []

    for item in feats_cfg:
        name = item.get("name")
        kind = item.get("kind")
        requires = tuple(item.get("requires", []) or [])
        params = item.get("params", {}) or {}

        if not name or not kind:
            raise ValueError(f"intra_features requires 'name' and 'kind': {item}")

        base_fn = INTRA_FEATURE_FUNCS.get(kind)
        if base_fn is None:
            raise ValueError(
                f"Unknown intra feature kind='{kind}'. Available: {sorted(INTRA_FEATURE_FUNCS)}"
            )

        # bind params into df -> float
        def make_func(fn, bound_params: dict):
            return lambda df: fn(df, **bound_params)

        out.append(
            {
                "name": name,
                "requires": requires,
                "func": make_func(base_fn, params),
            }
        )

    return out




def _per_ticker_daily(
    g: pd.DataFrame,
    *,
    ticker: str,
    cutoff_hour: float,
    market_tz: str,
    feature_specs: list[dict],
    daily_cfg: list[dict],
) -> pd.DataFrame:
    # assumes g has columns: timestamp, open, high, low, close, volume, ticker
    g = g.sort_values("timestamp")
    x = g.set_index("timestamp")  # required by aggregate_intraday_to_daily_features

    daily = aggregate_intraday_to_daily_features(
        x,
        cutoff_hour=cutoff_hour,
        tz=market_tz,
        features=feature_specs,
    )

    # make reset_index predictable
    daily.index.name = "timestamp"
    daily = daily.reset_index()
    daily["ticker"] = ticker

    # apply optional transforms
    if daily_cfg:
        daily = apply_transforms(daily, daily_cfg, DAILY_TRANSFORMS)

    return daily.sort_values("timestamp").reset_index(drop=True)


def build_gold_model_table(storage: Storage, paths: StoragePaths, gold_cfg: dict) -> pd.DataFrame:
    input_freq = gold_cfg.get("input_freq", "15Min")
    market_tz = gold_cfg.get("market_tz", "America/New_York")
    cutoff_hour = float(gold_cfg.get("cutoff_hour", 16.0))

    daily_cfg = gold_cfg.get("daily_transforms", []) or []
    assets = gold_cfg.get("assets", []) or []

    # Build once
    feature_specs = build_intra_feature_specs(gold_cfg)

    # 1) load intraday long
    df = load_multi_asset_flat_long(
        storage,
        paths,
        freq=input_freq,
        assets=assets,
        fields=("open", "high", "low", "close", "volume"),
        timestamp_col="timestamp",
        asset_col="ticker",
    )

    # normalize once
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["timestamp", "ticker"])
          .sort_values(["ticker", "timestamp"])
          .reset_index(drop=True)
    )

    frames: list[pd.DataFrame] = []
    for ticker, g in df.groupby("ticker", sort=False):
        frames.append(
            _per_ticker_daily(
                g,
                ticker=ticker,
                cutoff_hour=cutoff_hour,
                market_tz=market_tz,
                feature_specs=feature_specs,
                daily_cfg=daily_cfg,
            )
        )

    if not frames:
        return pd.DataFrame()

    gold = (
        pd.concat(frames, ignore_index=True)
          .sort_values(["ticker", "timestamp"])
          .reset_index(drop=True)
    )

    # write_gold_long_with_manifest(storage, paths, gold, gold_cfg=gold_cfg)
    return gold
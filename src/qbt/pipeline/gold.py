from __future__ import annotations

from typing import Sequence

import pandas as pd

from qbt.core.logging import get_logger
from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths
from qbt.storage.feature_store import write_gold_long_with_manifest

from qbt.features.aggregations import aggregate_intraday_to_daily_features
from qbt.features.apply import apply_transforms
from qbt.features.transforms import DAILY_TRANSFORMS
from qbt.features.intraday_transforms import INTRA_FEATURE_FUNCS
from qbt.data.loaders import load_multi_asset_flat_long
from qbt.utils.dates import stamp_asof_utc_from_session_dates

logger = get_logger(__name__)

def normalize_gold_cfg(gold_cfg: dict) -> dict:
    cfg = dict(gold_cfg or {})

    # defaults + type casting (do it once)
    cfg["input_freq"] = cfg.get("input_freq", "15Min")
    cfg["market_tz"] = cfg.get("market_tz", "America/New_York")

    cutoff = cfg.get("cutoff_hour", 16.0)
    cfg["cutoff_hour"] = float(cutoff)

    cfg["daily_transforms"] = cfg.get("daily_transforms", []) or []
    cfg["intraday_assets"] = cfg.get("intraday_assets", []) or []
    cfg["daily_assets"] = cfg.get("daily_assets", []) or []
    cfg["intra_features"] = cfg.get("intra_features", []) or []

    # light validation (optional but helpful)
    if not isinstance(cfg["daily_assets"], list):
        raise ValueError("gold_cfg.assets must be a list")
    if cfg["cutoff_hour"] <= 0 or cfg["cutoff_hour"] >= 24:
        raise ValueError("gold_cfg.cutoff_hour must be in (0, 24)")

    return cfg



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

def merge_daily_sources(
    storage: Storage,
    paths: StoragePaths,
    gold: pd.DataFrame,
    daily_assets: Sequence[str] | None = None,
) -> pd.DataFrame:

    # Load daily SILVER (freq = "1D")
    daily = load_multi_asset_flat_long(
        storage,
        paths,
        freq="1d",
        assets=daily_assets,
        timestamp_col="timestamp",
        asset_col="ticker",
    )

    if daily is None or daily.empty:
        return gold

    daily["session_date"] = (
        pd.to_datetime(daily["timestamp"], utc=True)
          .dt.tz_convert("UTC")
          .dt.tz_localize(None)
          .dt.normalize()
    )

    daily = daily.drop(columns=["timestamp"])

    # --- CRITICAL: make macro time-safe ---
    # shift by 1 day to avoid look-ahead
    daily = (
        daily.sort_values(["ticker", "session_date"])
             .groupby("ticker", as_index=False)
             .apply(lambda g: g.shift(1))
             .reset_index(drop=True)
    )

    merged = gold.merge(
        daily,
        on=["ticker", "session_date"],
        how="left",
        suffixes=("", "_daily"),
    )

    return merged


def _per_ticker_daily(
    g: pd.DataFrame,
    *,
    ticker: str,
    cutoff_hour: float,
    market_tz: str,
    feature_specs: list[dict],
    daily_cfg: list[dict],
) -> pd.DataFrame:

    g = g.sort_values("timestamp")
    x = g.set_index("timestamp")

    daily = aggregate_intraday_to_daily_features(
        x,
        cutoff_hour=cutoff_hour,
        tz=market_tz,
        features=feature_specs,
    )

    if daily.empty:
        return daily

    # Force session_date to tz-naive midnight labels
    idx = pd.to_datetime(daily.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx = idx.normalize()
    daily.index = pd.DatetimeIndex(idx, name="session_date")

    # Stamp cutoff as-of UTC
    daily["asof_utc"] = stamp_asof_utc_from_session_dates(
        daily.index,
        market_tz=market_tz,
        cutoff_hour=cutoff_hour,
    )

    daily = daily.reset_index()
    daily["ticker"] = ticker

    if daily_cfg:
        daily = apply_transforms(daily, daily_cfg, DAILY_TRANSFORMS)

    return daily.sort_values("session_date").reset_index(drop=True)


def build_gold_model_table(storage: Storage, paths: StoragePaths, gold_cfg: dict) -> pd.DataFrame:
    cfg = normalize_gold_cfg(gold_cfg)

    input_freq = cfg["input_freq"]
    market_tz = cfg["market_tz"]
    cutoff_hour = cfg["cutoff_hour"]

    daily_cfg = cfg["daily_transforms"]
    assets = cfg["intraday_assets"]
    daily_assets = cfg['daily_assets']
    feature_specs = build_intra_feature_specs(cfg)

    logger.info(
        f"Gold start | input_freq={input_freq} market_tz={market_tz} cutoff_hour={cutoff_hour} "
        f"assets={len(assets)} intra_features={len(cfg.get('intra_features', []) or [])} "
        f"daily_transforms={len(daily_cfg)}"
    )

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

    if df is None or df.empty:
        logger.warning("Gold: no intraday data loaded (empty)")
        return pd.DataFrame()

    logger.info(f"Loaded intraday long | rows={len(df)} cols={list(df.columns)}")

    # normalize once
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    before = len(df)
    df = (
        df.dropna(subset=["timestamp", "ticker"])
          .sort_values(["ticker", "timestamp"])
          .reset_index(drop=True)
    )
    dropped = before - len(df)
    if dropped:
        logger.info(f"Normalized intraday | dropped_rows={dropped} remaining={len(df)}")

    # compute per ticker
    frames: list[pd.DataFrame] = []
    for ticker, g in df.groupby("ticker", sort=False):
        t0 = g["timestamp"].min()
        t1 = g["timestamp"].max()
        logger.info(f"Compute daily features | ticker={ticker} rows={len(g)} t0={t0} t1={t1}")

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
        logger.warning("Gold: no per-ticker frames produced")
        return pd.DataFrame()

    gold = (
        pd.concat(frames, ignore_index=True)
          .sort_values(["ticker", "session_date"])
          .reset_index(drop=True)
    )

    gold = merge_daily_sources(storage, paths, gold, daily_assets=daily_assets)

    # small summary (avoid printing whole df)
    g0 = gold["session_date"].min() if "session_date" in gold.columns else None
    g1 = gold["session_date"].max() if "session_date" in gold.columns else None
    logger.info(f"Gold assembled | rows={len(gold)} cols={len(gold.columns)} session0={g0} session1={g1}")

    # write + manifest
    write_gold_long_with_manifest(storage, paths, gold, gold_cfg=gold_cfg)
    logger.info("Gold write complete")

    return gold
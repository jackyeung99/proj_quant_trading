from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from qbt.config.specs import DatasetSpec
from qbt.core.logging import get_logger
from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths
from qbt.storage.feature_store import write_gold_wide_with_manifest

from qbt.features.aggregations import aggregate_intraday_to_daily_features
from qbt.features.apply import apply_transforms
from qbt.features.transforms import DAILY_TRANSFORMS
from qbt.features.intraday_transforms import INTRA_FEATURE_FUNCS
from qbt.data.loaders import load_multi_asset_flat_long
from qbt.utils.dates import stamp_asof_utc_from_session_dates

logger = get_logger(__name__)


# remove this later 
def normalize_gold_spec(dataset: DatasetSpec) -> dict[str, Any]:
    aggregation = dict(dataset.aggregation or {})
    features = dict(dataset.features or {})
    output = dict(dataset.output or {})

    cfg: dict[str, Any] = {
        "dataset_name": dataset.name,
        "table_name": dataset.table_name,
        "input_freq": aggregation.get("input_freq", "15Min"),
        "market_tz": aggregation.get("market_tz", "America/New_York"),
        "cutoff_hour": float(aggregation.get("cutoff_hour", 16.0)),
        "daily_transforms": features.get("daily_transforms", []) or [],
        "intraday_assets": dataset.assets or [],
        "daily_assets": aggregation.get("daily_assets", []) or [],
        "intra_features": features.get("intraday", []) or [],
        "output": output,
    }

    if not isinstance(cfg["intraday_assets"], list):
        raise ValueError("dataset.assets must be a list")
    if not isinstance(cfg["daily_assets"], list):
        raise ValueError("dataset.aggregation.daily_assets must be a list")
    if cfg["cutoff_hour"] <= 0 or cfg["cutoff_hour"] >= 24:
        raise ValueError("cutoff_hour must be in (0, 24)")

    return cfg


def build_intra_feature_specs(intra_features_cfg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for item in intra_features_cfg:
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

        def make_func(fn, bound_params: dict[str, Any]):
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
    daily = load_multi_asset_flat_long(
        storage,
        paths,
        freq="1D",
        assets=daily_assets,
        timestamp_col="timestamp",
        fields=["close"],
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

    daily = (
        daily.sort_values(["ticker", "session_date"])
        .reset_index(drop=True)
        .dropna()
    )

    daily = (
        daily.pivot(index="session_date", columns="ticker", values="close")
        .reset_index()
    )

    merged = gold.merge(
        daily,
        on="session_date",
        how="left",
    )

    return merged


def _per_ticker_daily(
    g: pd.DataFrame,
    *,
    ticker: str,
    cutoff_hour: float,
    market_tz: str,
    feature_specs: list[dict[str, Any]],
    daily_cfg: list[dict[str, Any]],
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

    idx = pd.to_datetime(daily.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx = idx.normalize()
    daily.index = pd.DatetimeIndex(idx, name="session_date")

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


def build_gold_model_table(
    storage: Storage,
    paths: StoragePaths,
    dataset: DatasetSpec,
) -> pd.DataFrame:
    cfg = normalize_gold_spec(dataset)

    input_freq = cfg["input_freq"]
    market_tz = cfg["market_tz"]
    cutoff_hour = cfg["cutoff_hour"]

    daily_cfg = cfg["daily_transforms"]
    assets = cfg["intraday_assets"]
    daily_assets = cfg["daily_assets"]
    feature_specs = build_intra_feature_specs(cfg["intra_features"])

    logger.info(
        "Gold start | dataset=%s table=%s input_freq=%s market_tz=%s cutoff_hour=%s "
        "assets=%d intra_features=%d daily_transforms=%d",
        dataset.name,
        dataset.table_name,
        input_freq,
        market_tz,
        cutoff_hour,
        len(assets),
        len(cfg["intra_features"]),
        len(daily_cfg),
    )

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

    logger.info("Loaded intraday long | rows=%d cols=%s", len(df), list(df.columns))

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    before = len(df)
    df = (
        df.dropna(subset=["timestamp", "ticker"])
        .sort_values(["ticker", "timestamp"])
        .reset_index(drop=True)
    )
    dropped = before - len(df)
    if dropped:
        logger.info("Normalized intraday | dropped_rows=%d remaining=%d", dropped, len(df))

    frames: list[pd.DataFrame] = []
    for ticker, g in df.groupby("ticker", sort=False):
        t0 = g["timestamp"].min()
        t1 = g["timestamp"].max()
        logger.info(
            "Compute daily features | ticker=%s rows=%d t0=%s t1=%s",
            ticker,
            len(g),
            t0,
            t1,
        )

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

    gold_long = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "session_date"])
        .reset_index(drop=True)
    )

    wide = (
        gold_long.reset_index()
        .melt(
            id_vars=["session_date", "ticker"],
            var_name="field",
            value_name="value",
        )
        .pivot_table(
            index="session_date",
            columns=["ticker", "field"],
            values="value",
            aggfunc="last",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    wide = wide.copy().sort_index().sort_index(axis=1, level=[0, 1])
    wide.columns = [f"{asset}_{field}" for asset, field in wide.columns]

    gold = merge_daily_sources(storage, paths, wide, daily_assets=daily_assets)

    g0 = gold["session_date"].min() if "session_date" in gold.columns else None
    g1 = gold["session_date"].max() if "session_date" in gold.columns else None
    logger.info(
        "Gold assembled | rows=%d cols=%d session0=%s session1=%s",
        len(gold),
        len(gold.columns),
        g0,
        g1,
    )

    write_gold_wide_with_manifest(
        storage,
        paths,
        gold,
        dataset=dataset,
    )
    logger.info("Gold write complete")

    return gold
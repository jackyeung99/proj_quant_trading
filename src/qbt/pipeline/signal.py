from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Any

import pandas as pd

from qbt.core.logging import get_logger
from qbt.core.types import RunSpec, ModelInputs, ModelBundle
from qbt.data.dataloader import DataAdapter, WidePrefixDataAdapter
from qbt.strategies.strategy_registry import create_strategy
from qbt.utils.dates import _to_utc, _session_date_from_ts, _stamp_asof_utc
from qbt.utils.stamping import make_snapshot_id, config_hash

logger = get_logger(__name__)

def should_retrain(
    now_iso: str,
    meta: dict | None,
    *,
    retrain_freq: str,
    market_tz: str = "America/New_York",
    config_hash_now: str | None = None,
) -> bool:
    if meta is None:
        return True

    # accept either key; prefer the explicit UTC one
    trained_key = "trained_at_utc" if "trained_at_utc" in meta else "trained_at"
    if trained_key not in meta or meta[trained_key] in (None, "", "unknown"):
        return True

    if config_hash_now is not None and meta.get("config_hash") not in (None, config_hash_now):
        return True

    freq = retrain_freq
    if freq in (None, "none", "None", False):
        return False
    if freq == "always":
        return True

    trained_at_utc = _to_utc(meta[trained_key])
    now_utc = _to_utc(now_iso)

    # Compare in MARKET calendar (not UTC calendar)
    if freq == "1D":
        return trained_at_utc.tz_convert(market_tz).date() != now_utc.tz_convert(market_tz).date()

    if freq == "1W":
        ta = trained_at_utc.tz_convert(market_tz).isocalendar()
        na = now_utc.tz_convert(market_tz).isocalendar()
        return (ta.year, ta.week) != (na.year, na.week)

    try:
        return now_utc >= trained_at_utc + pd.Timedelta(freq)
    except Exception as e:
        raise ValueError(f"Unknown retrain_freq={freq!r}") from e


def prepare_inputs(
    data_adapter: DataAdapter,
    spec: RunSpec,
    required_asset_features: list[str],
    required_global_features: list[str],
) -> ModelInputs:
    raw = data_adapter.load(spec)
    return data_adapter.prepare(
        raw,
        spec,
        required_asset_features=required_asset_features,
        required_global_features=required_global_features,
        assets=getattr(spec, "assets", None),
    )


def _slice_model_inputs(inputs: ModelInputs, idx: pd.Index) -> ModelInputs:
    return ModelInputs(
        ret=inputs.ret.loc[idx],
        asset_features={
            name: panel.loc[idx]
            for name, panel in inputs.asset_features.items()
        },
        global_features=inputs.global_features.loc[idx],
    )


def slice_train_window(inputs: ModelInputs, *, lookback: int) -> ModelInputs:
    idx = inputs.ret.index.sort_values()
    idx = idx[-lookback:]
    return _slice_model_inputs(inputs, idx)


def _model_input_length(inputs: ModelInputs) -> int:
    return len(inputs.ret.index)


def enforce_bundle_schema(bundle: ModelBundle, inputs: ModelInputs) -> ModelInputs:
    ret = inputs.ret.copy()

    missing_ret = [c for c in bundle.ret_cols if c not in ret.columns]
    if missing_ret:
        raise ValueError(f"Missing return columns at inference: {missing_ret}")

    ret = ret.reindex(columns=bundle.ret_cols)

    required_asset_features = list(getattr(bundle, "asset_feature_cols", []) or [])
    required_global_features = list(getattr(bundle, "global_feature_cols", []) or [])

    missing_asset_features = [
        feat for feat in required_asset_features
        if feat not in inputs.asset_features
    ]
    if missing_asset_features:
        raise ValueError(f"Missing asset features at inference: {missing_asset_features}")

    asset_features = {}
    for feat in required_asset_features:
        panel = inputs.asset_features[feat]
        missing_assets = [c for c in bundle.ret_cols if c not in panel.columns]
        if missing_assets:
            raise ValueError(
                f"Asset feature '{feat}' missing assets at inference: {missing_assets}"
            )
        asset_features[feat] = panel.reindex(columns=bundle.ret_cols)

    missing_global = [
        c for c in required_global_features
        if c not in inputs.global_features.columns
    ]
    if missing_global:
        raise ValueError(f"Missing global features at inference: {missing_global}")

    global_features = inputs.global_features.reindex(columns=required_global_features)

    idx = ret.index
    for feat, panel in asset_features.items():
        idx = idx.intersection(panel.index)
    idx = idx.intersection(global_features.index)

    ret = ret.loc[idx]
    asset_features = {k: v.loc[idx] for k, v in asset_features.items()}
    global_features = global_features.loc[idx]

    return ModelInputs(
        ret=ret,
        asset_features=asset_features,
        global_features=global_features,
    )


def latest_row_df(w: pd.DataFrame) -> pd.DataFrame:
    if w is None or not isinstance(w, pd.DataFrame) or w.empty:
        raise ValueError("Strategy must return a non-empty DataFrame of weights.")
    if not isinstance(w.index, pd.DatetimeIndex):
        raise ValueError("Weights DataFrame must be indexed by DatetimeIndex.")
    return w.tail(1).astype(float).fillna(0.0)

def latest_row_df(w: pd.DataFrame) -> pd.DataFrame:
    if w is None or not isinstance(w, pd.DataFrame) or w.empty:
        raise ValueError("Strategy must return a non-empty DataFrame of weights.")
    if not isinstance(w.index, pd.DatetimeIndex):
        raise ValueError("Weights DataFrame must be indexed by DatetimeIndex.")
    return w.tail(1).astype(float).fillna(0.0)


def signal(
    live_storage,
    strat_cfg: dict,
    *,
    data_adapter: Optional[DataAdapter] = None,
) -> pd.DataFrame:
    cfg = strat_cfg or {}
    run_cfg = cfg.get("run", {}) or {}
    live_cfg = cfg.get("live", {}) or {}

    spec = RunSpec(**run_cfg)
    universe = getattr(spec, "universe", None) or run_cfg.get("universe") or run_cfg.get("ticker") or "default"

    market_tz = run_cfg.get("market_tz", "America/New_York")
    cutoff_hour = float(run_cfg.get("cutoff_hour", 16.0))

    retrain_freq = live_cfg.get("retrain_freq", "1D")
    lookback = int(live_cfg.get("train_lookback_bars", 252))
    min_train = int(live_cfg.get("min_train_bars", 200))

    now_utc = pd.Timestamp.now(tz="UTC")
    now_iso = now_utc.isoformat()

    cfg_hash = config_hash(run_cfg)

    data_adapter = data_adapter or WidePrefixDataAdapter(storage=live_storage.storage)

    strat = create_strategy(spec.strategy_name)
    req_asset = strat.required_asset_features(spec)
    req_global = strat.required_global_features(spec)

    logger.info(
        f"Signal start | strategy={spec.strategy_name} universe={universe} "
        f"retrain_freq={retrain_freq} lookback={lookback} min_train={min_train} market_tz={market_tz}"
    )

    bundle = live_storage.read_model(spec.strategy_name, universe)
    meta = live_storage.read_model_meta(spec.strategy_name, universe) or None

    do_train = should_retrain(
        now_iso,
        meta,
        retrain_freq=retrain_freq,
        market_tz=market_tz,
        config_hash_now=cfg_hash,
    )

    snapshot_id: str | None = None

    if do_train:
        logger.info("Retrain triggered")

        inputs_all = prepare_inputs(
            data_adapter,
            spec,
            required_asset_features=req_asset,
            required_global_features=req_global,
        )
    
        train_inputs = slice_train_window(inputs_all, lookback=lookback)

        n_train = _model_input_length(train_inputs)
        if n_train < min_train:
            raise ValueError(f"Not enough train data: {n_train} < {min_train}")

        strat.fit(train_inputs, spec)
        logger.info("Model fit complete")

        train_start_session = pd.Timestamp(train_inputs.ret.index.min()).normalize()
        train_end_session = pd.Timestamp(train_inputs.ret.index.max()).normalize()

        train_start_asof_utc = _stamp_asof_utc(train_start_session, market_tz=market_tz, hour=cutoff_hour)
        train_end_asof_utc = _stamp_asof_utc(train_end_session, market_tz=market_tz, hour=cutoff_hour)

        trained_at_utc = now_iso
        snapshot_id = make_snapshot_id(trained_at_utc, cfg_hash)

        bundle = ModelBundle(
            model=strat,
            asset_feature_cols=list(req_asset),
            global_feature_cols=list(req_global),
            ret_cols=list(train_inputs.ret.columns),
            trained_at=trained_at_utc,
            train_start=str(train_start_asof_utc),
            train_end=str(train_end_asof_utc),
            config_hash=cfg_hash,
        )

        meta_out = {
            "snapshot_id": snapshot_id,
            "trained_at_utc": trained_at_utc,
            "config_hash": cfg_hash,
            "train_start_session_date": str(train_start_session),
            "train_end_session_date": str(train_end_session),
            "train_start_asof_utc": str(train_start_asof_utc),
            "train_end_asof_utc": str(train_end_asof_utc),
            "retrain_freq": retrain_freq,
            "train_lookback_bars": lookback,
            "min_train_bars": min_train,
            "asset_feature_cols": list(req_asset),
            "global_feature_cols": list(req_global),
            "ret_cols": list(train_inputs.ret.columns),
            "bundle_version": getattr(bundle, "version", "v1"),
            "market_tz": market_tz,
            "cutoff_hour": cutoff_hour,
            "strategy_meta": strat.get_model_state(),
        }

        live_storage.write_model(
            strategy=spec.strategy_name,
            universe=universe,
            bundle=bundle,
            meta=meta_out,
            snapshot=bool(live_cfg.get("snapshot_models", True)),
        )

        meta = meta_out
        logger.info(f"Wrote model latest + snapshot_id={snapshot_id}")
    else:
        logger.info("Retrain skipped")
        if meta:
            snapshot_id = meta.get("snapshot_id")

    if bundle is None:
        raise RuntimeError("No trained live model available (missing model artifact/meta).")

    inputs_now = prepare_inputs(
        data_adapter,
        spec,
        required_asset_features=list(getattr(bundle, "asset_feature_cols", []) or []),
        required_global_features=list(getattr(bundle, "global_feature_cols", []) or []),
    )
    inputs_now = enforce_bundle_schema(bundle, inputs_now)

    last_idx = pd.Timestamp(inputs_now.ret.index.max())
    session_date = last_idx.normalize()
    asof_utc = _stamp_asof_utc(session_date, market_tz=market_tz, hour=cutoff_hour)

    w_ts = bundle.model.predict(inputs_now, spec)
    latest_w = latest_row_df(w_ts)

    latest_w.index = pd.DatetimeIndex([session_date], name="session_date")
    latest_w["session_date"] = session_date
    latest_w["asof_utc"] = asof_utc
    latest_w["generated_at_utc"] = now_utc
    latest_w["config_hash"] = cfg_hash
    latest_w["market_tz"] = market_tz
    latest_w["cutoff_hour"] = cutoff_hour

    if snapshot_id is None and meta:
        snapshot_id = meta.get("snapshot_id")
    latest_w["snapshot_id"] = snapshot_id or "unknown"

    live_storage.write_weights(
        strategy=spec.strategy_name,
        universe=universe,
        latest_w=latest_w,
        snapshot=bool(live_cfg.get("snapshot_weights", True)),
    )

    return latest_w
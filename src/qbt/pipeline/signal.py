from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Any

import pandas as pd

from qbt.core.logging import get_logger
from qbt.core.types import RunSpec, ModelInputs, ModelBundle
from qbt.data.dataloader import DataAdapter, DefaultDataAdapter
from qbt.strategies.strategy_registry import create_strategy
from qbt.utils.config_parser import config_hash
from qbt.utils.dates import _to_utc, _session_date_from_ts, _stamp_asof_utc

logger = get_logger(__name__)

def should_retrain(
    now_iso: str,
    meta: dict | None,
    *,
    retrain_freq: str,
    market_tz: str = "America/New_York",
    config_hash_now: str | None = None,
) -> bool:
    if meta is None or "trained_at" not in meta:
        return True

    if config_hash_now is not None and meta.get("config_hash") not in (None, config_hash_now):
        return True

    freq = retrain_freq
    if freq in (None, "none", "None", False):
        return False
    if freq == "always":
        return True

    trained_at_utc = _to_utc(meta["trained_at"])
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
    required_cols: list[str],
) -> ModelInputs:
    raw = data_adapter.load(spec)
    return data_adapter.prepare(raw, spec, required_cols=required_cols)


def slice_train_window(inputs: ModelInputs, *, lookback: int) -> ModelInputs:
    idx = inputs.features.index.intersection(inputs.ret.index)
    feats = inputs.features.loc[idx].tail(lookback)
    rets = inputs.ret.loc[feats.index]
    return ModelInputs(ret=rets, features=feats)


def enforce_bundle_schema(bundle: ModelBundle, inputs: ModelInputs) -> ModelInputs:
    idx = inputs.features.index.intersection(inputs.ret.index)
    feats = inputs.features.loc[idx]

    missing = [c for c in bundle.feature_cols if c not in feats.columns]
    if missing:
        raise ValueError(f"Missing feature columns at inference: {missing}")

    feats = feats.reindex(columns=bundle.feature_cols)
    rets = inputs.ret.loc[idx].reindex(columns=bundle.ret_cols)

    return ModelInputs(ret=rets, features=feats)


def latest_row_df(w: pd.DataFrame) -> pd.DataFrame:
    if w is None or not isinstance(w, pd.DataFrame) or w.empty:
        raise ValueError("Strategy must return a non-empty DataFrame of weights.")
    if not isinstance(w.index, pd.DatetimeIndex):
        raise ValueError("Weights DataFrame must be indexed by DatetimeIndex.")
    return w.tail(1).astype(float).fillna(0.0)


def signal(
    live_storage,  # LiveStore
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
    cutoff_hour = float(run_cfg.get("cutoff_hour", 16.0))  # used only for stamping/audit

    retrain_freq = live_cfg.get("retrain_freq", "1D")
    lookback = int(live_cfg.get("train_lookback_bars", 252))
    min_train = int(live_cfg.get("min_train_bars", 200))

    now_utc = pd.Timestamp.now(tz="UTC")
    now_iso = now_utc.isoformat()

    cfg_hash = config_hash(run_cfg)

    # live store has connection to live_storage , fix later 
    data_adapter = DefaultDataAdapter(storage=live_storage.storage)

    strat = create_strategy(spec.strategy_name)
    req = strat.required_features(spec)

    logger.info(
        f"Signal start | strategy={spec.strategy_name} universe={universe} "
        f"retrain_freq={retrain_freq} lookback={lookback} min_train={min_train} market_tz={market_tz}"
    )

    # Load existing live artifacts
    bundle = live_storage.read_model(spec.strategy_name, universe)
    meta = live_storage.read_model_meta(spec.strategy_name, universe) or None

    do_train = should_retrain(
        now_iso,
        meta,
        retrain_freq=retrain_freq,
        market_tz=market_tz,
        config_hash_now=cfg_hash,
    )

    # -------------------------
    # TRAIN (optional)
    # -------------------------
    if do_train:
        logger.info("Retrain triggered")

        inputs_all = prepare_inputs(data_adapter, spec, required_cols=req)
        train_inputs = slice_train_window(inputs_all, lookback=lookback)

        n_train = len(train_inputs.features)
        if n_train < min_train:
            raise ValueError(f"Not enough train data: {n_train} < {min_train}")

        strat.fit(train_inputs, spec)
        logger.info("Model fit complete")

        train_end_ts = pd.Timestamp(train_inputs.features.index.max())
        train_end_session = _session_date_from_ts(train_end_ts, market_tz=market_tz)

        bundle = ModelBundle(
            model=strat,
            feature_cols=list(train_inputs.features.columns),
            ret_cols=list(train_inputs.ret.columns),
            trained_at=now_iso,  # UTC iso
            train_end=str(train_end_session),  # store as session_date string (stable)
            config_hash=cfg_hash,
        )

        meta_out = {
            "trained_at": bundle.trained_at,   # UTC iso
            "train_end": bundle.train_end,     # session_date string
            "train_end_session_date": str(train_end_session),
            "retrain_freq": retrain_freq,
            "train_lookback_bars": lookback,
            "min_train_bars": min_train,
            "feature_cols": bundle.feature_cols,
            "ret_cols": bundle.ret_cols,
            "config_hash": bundle.config_hash,
            "bundle_version": getattr(bundle, "version", "v1"),
            "market_tz": market_tz,
            "cutoff_hour": cutoff_hour,
            "strategy_meta": strat.get_meta(),
        }

        live_storage.write_model(
            strategy=spec.strategy_name,
            universe=universe,
            bundle=bundle,
            meta=meta_out,
            snapshot=bool(live_cfg.get("snapshot_models", True)),
        )
        logger.info(
            f"Wrote model latest + snapshot={bool(live_cfg.get('snapshot_models', True))} "
            f"trained_at={bundle.trained_at}"
        )
    else:
        logger.info("Retrain skipped")

    if bundle is None:
        raise RuntimeError("No trained live model available (missing model artifact/meta).")

    # -------------------------
    # INFERENCE
    # -------------------------
    inputs_now = prepare_inputs(data_adapter, spec, required_cols=req)
    inputs_now = enforce_bundle_schema(bundle, inputs_now)

    last_feat_ts = pd.Timestamp(inputs_now.features.index.max())
    session_date = _session_date_from_ts(last_feat_ts, market_tz=market_tz)

    # deterministic "asof at cutoff time (e.g., 16:00 ET)" stamp in UTC
    asof_utc = _stamp_asof_utc(session_date, market_tz=market_tz, hour=cutoff_hour)

    w_ts = bundle.model.predict(inputs_now, spec)
    latest_w = latest_row_df(w_ts)

    # index weights by session_date for merges
    latest_w.index = pd.DatetimeIndex([session_date], name="session_date")

    # stamps
    latest_w["asof_utc"] = asof_utc
    latest_w["generated_at_utc"] = now_utc
    latest_w["config_hash"] = cfg_hash
    latest_w["market_tz"] = market_tz
    latest_w["cutoff_hour"] = cutoff_hour

    # -------------------------
    # WRITE WEIGHTS (latest + snapshot)
    # -------------------------
    live_storage.write_weights(
        strategy=spec.strategy_name,
        universe=universe,
        latest_w=latest_w,
        snapshot=bool(live_cfg.get("snapshot_weights", True)),
    )

    return latest_w
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from qbt.core.types import RunSpec, ModelInputs, ModelBundle
from qbt.data.dataloader import DataAdapter, DefaultDataAdapter
from qbt.strategies.strategy_registry import create_strategy
from qbt.utils.config_parser import config_hash


# =============================================================================
# Retrain policy
# =============================================================================

def should_retrain(
    now_iso: str,
    meta: dict | None,
    *,
    retrain_freq: str,
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

    trained_at = pd.Timestamp(meta["trained_at"])
    now = pd.Timestamp(now_iso)

    if freq == "1D":
        return trained_at.date() != now.date()
    if freq == "1W":
        ta = trained_at.isocalendar()
        na = now.isocalendar()
        return (ta.year, ta.week) != (na.year, na.week)

    try:
        return now >= trained_at + pd.Timedelta(freq)
    except Exception as e:
        raise ValueError(f"Unknown retrain_freq={freq!r}") from e


# =============================================================================
# Input utilities
# =============================================================================

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


# =============================================================================
# Weights utilities (DF-only)
# =============================================================================

def latest_row_df(w: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce: weights are a time-indexed DataFrame (DatetimeIndex),
    columns are assets, return 1-row DF.
    """
    if w is None or not isinstance(w, pd.DataFrame) or w.empty:
        raise ValueError("Strategy must return a non-empty DataFrame of weights.")
    if not isinstance(w.index, pd.DatetimeIndex):
        raise ValueError("Weights DataFrame must be indexed by DatetimeIndex.")
    return w.tail(1).astype(float).fillna(0.0)


# =============================================================================
# Main step (live)
# =============================================================================

def signal(
    storage,  # <-- LiveStore (or LiveArtifactStore)
    strat_cfg: dict,
    *,
    data_adapter: Optional[DataAdapter] = None,
) -> pd.DataFrame:
    """
    Live signal step (idempotent):
      - Read live model/meta from storage (LiveStore)
      - Retrain if policy says so; persist model/meta (and optional snapshot)
      - Predict weights (DF, time-indexed)
      - Append latest weights via storage.append_weights() (idempotent upsert by asof)

    Returns: single-row DataFrame indexed by asof with latest weights (+generated_at_utc, config_hash)
    """
    cfg = strat_cfg or {}
    run_cfg = cfg.get("run", {}) or {}
    live_cfg = cfg.get("live", {}) or {}

    spec = RunSpec(**run_cfg)

    # universe is used as the live namespace (XLE, XLE-SPY, etc.)
    # if RunSpec has it, use that; else fall back.
    universe = getattr(spec, "universe", None) or run_cfg.get("universe") or run_cfg.get("ticker") or "default"


    retrain_freq = live_cfg.get("retrain_freq", "1D")
    lookback = int(live_cfg.get("train_lookback_bars", 252))
    min_train = int(live_cfg.get("min_train_bars", 200))

    now_iso = datetime.now(timezone.utc).isoformat()
    cfg_hash = config_hash(run_cfg)

    data_adapter = data_adapter or DefaultDataAdapter()

    # Build strategy once (needed for required features in both branches)
    strat = create_strategy(spec.strategy_name)
    req = strat.required_features(spec)

    # Load existing live artifacts
    bundle = storage.read_model(spec.strategy_name, universe)
    meta = storage.read_model_meta(spec.strategy_name, universe) or None

    do_train = should_retrain(
        now_iso,
        meta,
        retrain_freq=retrain_freq,
        config_hash_now=cfg_hash,
    )

    if do_train:
        inputs_all = prepare_inputs(data_adapter, spec, required_cols=req)
        train_inputs = slice_train_window(inputs_all, lookback=lookback)

        if len(train_inputs.features) < min_train:
            raise ValueError(f"Not enough train data: {len(train_inputs.features)} < {min_train}")

        strat.fit(train_inputs, spec)

        bundle = ModelBundle(
            model=strat,
            feature_cols=list(train_inputs.features.columns),
            ret_cols=list(train_inputs.ret.columns),
            trained_at=now_iso,
            train_end=str(train_inputs.features.index.max()),
            config_hash=cfg_hash,
        )

        meta_out = {
            "trained_at": bundle.trained_at,
            "train_end": bundle.train_end,
            "retrain_freq": retrain_freq,
            "train_lookback_bars": lookback,
            "min_train_bars": min_train,
            "feature_cols": bundle.feature_cols,
            "ret_cols": bundle.ret_cols,
            "config_hash": bundle.config_hash,
            "bundle_version": getattr(bundle, "version", "v1"),
        }

        storage.write_model(
            strategy=spec.strategy_name,
            universe=universe,
            bundle=bundle,
            meta=meta_out,
            snapshot=bool(live_cfg.get("snapshot_models", True)),
        )

    if bundle is None:
        raise RuntimeError("No trained live model available (missing model artifact/meta).")

    # Predict using latest available inputs; enforce schema from bundle
    inputs_now = prepare_inputs(data_adapter, spec, required_cols=req)
    inputs_now = enforce_bundle_schema(bundle, inputs_now)

    w_ts = bundle.model.predict(inputs_now, spec)  # EXPECTED: time-indexed DF (assets as columns)

    latest_w = latest_row_df(w_ts)

    # align asof to actual aligned inputs index
    idx_now = inputs_now.features.index.intersection(inputs_now.ret.index)
    asof_ts = pd.Timestamp(idx_now.max()) if len(idx_now) else pd.Timestamp(now_iso)

    latest_w.index = pd.DatetimeIndex([asof_ts], name="asof")
    latest_w["generated_at_utc"] = now_iso
    latest_w["config_hash"] = cfg_hash

    # Persist weights via live store (idempotent upsert by asof)
    storage.write_weights(
        strategy=spec.strategy_name,
        universe=universe,
        latest_w=latest_w,
    )

    return latest_w

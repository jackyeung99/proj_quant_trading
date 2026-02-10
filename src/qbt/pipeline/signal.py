from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths

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
    """
    Idempotent retrain decision.

    Retrain if:
      - no meta
      - config hash changed (optional)
      - retrain_freq says we should
    """
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

    # also allow pandas-style timedeltas: "6H", "30min", etc.
    try:
        return now >= trained_at + pd.Timedelta(freq)
    except Exception as e:
        raise ValueError(f"Unknown retrain_freq={freq!r}") from e


# =============================================================================
# Weights utilities
# =============================================================================

def normalize_latest_weights(w: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Return a single-row DataFrame of latest weights."""
    if isinstance(w, pd.Series):
        out = w.to_frame().T
    else:
        if w is None or w.empty:
            raise ValueError("Strategy returned empty weights DataFrame.")
        out = w.tail(1)

    return out.astype(float).fillna(0.0)


def append_idempotent_timeseries(
    storage: Storage,
    key: str,
    row: pd.DataFrame,
    *,
    index_name: str = "asof",
) -> pd.DataFrame:
    """
    Append a single-row DF into a parquet timeseries, idempotently:
    - union columns
    - drop existing rows with same index
    - sort by index
    """
    if row.shape[0] != 1:
        raise ValueError("append_idempotent_timeseries expects a single-row DataFrame.")

    row = row.copy()
    row.index.name = index_name
    ts = row.index[0]

    if storage.exists(key):
        prev = storage.read_parquet(key)
        prev.index.name = index_name

        all_cols = prev.columns.union(row.columns)
        prev = prev.reindex(columns=all_cols)
        row = row.reindex(columns=all_cols)

        # idempotent overwrite of the same timestamp
        prev = prev.loc[prev.index != ts]
        out = pd.concat([prev, row]).sort_index()
    else:
        out = row

    storage.write_parquet(out, key)
    return out


# =============================================================================
# Training utilities
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
# Main step
# =============================================================================

def signal(
    storage: Storage,
    paths: StoragePaths,
    strat_cfg: dict,
    *,
    data_adapter: Optional[DataAdapter] = None,
) -> pd.DataFrame:
    """
    Live signal step (idempotent):
      - Decide whether to retrain using (meta, retrain_freq, config_hash)
      - Train and persist bundle/meta if needed
      - Predict weights for latest slice
      - Append latest weights to weights parquet idempotently (upsert by asof timestamp)

    Returns: single-row DataFrame with latest weights (+ generated_at_utc)
    """
    cfg = strat_cfg or {}
    run_cfg = cfg.get("run", {}) or {}
    live_cfg = cfg.get("live", {}) or {}

    spec = RunSpec(**run_cfg)

    retrain_freq = live_cfg.get("retrain_freq", "1D")
    lookback = int(live_cfg.get("train_lookback_bars", 252))
    min_train = int(live_cfg.get("min_train_bars", 200))

    model_key = paths.model_key(spec.strategy_name)
    meta_key = paths.model_meta_key(spec.strategy_name)
    latest_w_key =  paths.latest_weight_key(spec.strategy_name)

    now_iso = datetime.now(timezone.utc).isoformat()
    cfg_hash = config_hash(run_cfg)

    # load existing artifacts
    bundle: Optional[ModelBundle] = None
    meta: Optional[dict] = None
    if storage.exists(model_key) and storage.exists(meta_key):
        bundle = storage.read_pickle(model_key)
        meta = storage.read_json(meta_key)

    do_train = should_retrain(
        now_iso,
        meta,
        retrain_freq=retrain_freq,
        config_hash_now=cfg_hash,
    )

    data_adapter = data_adapter or DefaultDataAdapter()

    # build strategy once; safe to use for required_features in both branches
    strat = create_strategy(spec.strategy_name)
    req = strat.required_features(spec)

    if do_train:
        inputs_all = prepare_inputs(data_adapter, spec, required_cols=req)
        train_inputs = slice_train_window(inputs_all, lookback=lookback)

        if len(train_inputs.features) < min_train:
            raise ValueError(
                f"Not enough train data: {len(train_inputs.features)} < {min_train}"
            )

        strat.fit(train_inputs, spec)

        bundle = ModelBundle(
            model=strat,
            feature_cols=list(train_inputs.features.columns),
            ret_cols=list(train_inputs.ret.columns),
            trained_at=now_iso,
            train_end=str(train_inputs.features.index.max()),
            config_hash=cfg_hash,
        )

        storage.write_pickle(bundle, model_key)
        storage.write_json(
            {
                "trained_at": bundle.trained_at,
                "train_end": bundle.train_end,
                "retrain_freq": retrain_freq,
                "train_lookback_bars": lookback,
                "min_train_bars": min_train,
                "feature_cols": bundle.feature_cols,
                "ret_cols": bundle.ret_cols,
                "config_hash": bundle.config_hash,
                "bundle_version": getattr(bundle, "version", "v1"),
            },
            meta_key
        )

    if bundle is None:
        raise RuntimeError("No trained model available (missing model artifact/meta).")

    # Predict using latest available inputs; enforce schema from bundle
    inputs_now = prepare_inputs(data_adapter, spec, required_cols=req)
    inputs_now = enforce_bundle_schema(bundle, inputs_now)

    w = bundle.model.predict(inputs_now, spec)

    latest_w = normalize_latest_weights(w)

    # asof timestamp: prefer the model inputs index max
    idx_now = inputs_now.features.index.intersection(inputs_now.ret.index)
    asof_ts = pd.Timestamp(idx_now.max()) if len(idx_now) else pd.Timestamp(now_iso)

    latest_w.index = pd.Index([asof_ts], name="asof")
    latest_w["generated_at_utc"] = now_iso
    latest_w["config_hash"] = cfg_hash

    # Idempotent append/upsert by asof timestamp
    append_idempotent_timeseries(storage, latest_w_key, latest_w, index_name="asof")

    return latest_w

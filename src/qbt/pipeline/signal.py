
import yaml
import pandas as pd
from pathlib import Path

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths

from qbt.core.types import RunSpec, RunMeta, RunResult, BacktestSpec, ModelInputs
from qbt.data.dataloader import DataAdapter, DefaultDataAdapter

from qbt.strategies.strategy_registry import create_strategy, available_strategies
from qbt.strategies.strategy_base import Strategy


def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def should_retrain(now: pd.Timestamp, meta: dict | None, params: dict) -> bool:
    if meta is None:
        return True

    freq = params.get("retrain_freq", "1D")  # "1D", "1W", "always", "none"
    if freq in (None, "none", "None", False):
        return False
    if freq == "always":
        return True

    trained_at = pd.Timestamp(meta["trained_at"])

    if freq == "1D":
        return trained_at.date() != now.date()
    if freq == "1W":
        ta = trained_at.isocalendar()
        na = now.isocalendar()
        return (ta.year, ta.week) != (na.year, na.week)

    raise ValueError(f"Unknown retrain_freq={freq}")

def signal(storage: Storage, paths: StoragePaths, cfg: dict, now: pd.Timestamp):
    strat_cfg = load_yaml(cfg["decision"]["model"])
    spec = RunSpec(**strat_cfg)

    strat = create_strategy(spec.strategy_name)
    req = strat.required_features(spec)

    prefix = model_prefix(spec)
    model_key = f"{prefix}/model.pkl"
    meta_key  = f"{prefix}/meta.json"

    fitted = None
    meta = None
    if storage.exists(model_key) and storage.exists(meta_key):
        fitted = storage.read_pickle(model_key)   # your Storage should expose this
        meta = storage.read_json(meta_key)

    do_train = should_retrain(now, meta, spec.params)

    data_adapter = DefaultDataAdapter()

    if do_train:
        # IMPORTANT: only use data up to cutoff
        # best: adapter supports pull_end=now (or cutoff_ts)
        raw = data_adapter.load(spec, pull_end=now)
        inputs: ModelInputs = data_adapter.prepare(raw, spec, required_cols=req)

        lookback = int(spec.params.get("train_lookback_bars", 252))
        min_train = int(spec.params.get("min_train_bars", 200))
        train = inputs.tail(lookback)
        if len(train) < min_train:
            raise ValueError(f"Not enough train data: {len(train)} < {min_train}")

        strat.fit(train, spec)

        storage.write_pickle(model_key, strat)
        storage.write_json(meta_key, {
            "trained_at": str(now),
            "train_end": str(train.index.max()),
            "lookback_bars": lookback,
        })
    else:
        # reuse fitted artifact
        strat = fitted
        if strat is None:
            # safety: if meta exists but model missing
            raise RuntimeError("Model meta exists but model artifact missing.")

    # Predict using the most recent available slice (again cutoff-aware)
    raw_now = data_adapter.load(spec, pull_end=now)
    inputs_now = data_adapter.prepare(raw_now, spec, required_cols=req)

    w = strat.predict(inputs_now, spec)
    return w

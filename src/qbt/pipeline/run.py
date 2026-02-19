from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths

from qbt.pipeline.ingestion import ingest_all_sources
from qbt.pipeline.silver import canonicalize_all
from qbt.pipeline.gold import build_gold_model_table
from qbt.pipeline.execute import execute_weights
from qbt.pipeline.signal import signal
from qbt.pipeline.evaluation import evaluate_portfolio

import pprint

def run_pipeline(storage, paths, cfg, artifact_store):


    if cfg["ingestion"]["enabled"]:
        ingest_all_sources(
            storage,
            paths,
            ingestion_cfg=cfg["ingestion"]['cfg'],
            sources_cfg=cfg["sources"],
        )

    if cfg["silver"]["enabled"]:
        canonicalize_all(
            storage,
            paths,
            sources_cfg=cfg["sources"],
        )

    if cfg["gold"]["enabled"]:
        build_gold_model_table(
            storage,
            paths,
            gold_cfg=cfg["gold"]['cfg'],
        )

    if cfg["signal"]["enabled"]:
        signal(
            storage=artifact_store,
            # paths=paths,
            strat_cfg=cfg["signal"]['cfg'],
        )

    if cfg["execution"]["enabled"]:
        execute_weights(
            artifact_store,
            # paths,
            execution_cfg=cfg["execution"]['cfg'],
        )

    if cfg["evaluation"]["enabled"]:
        evaluate_portfolio(
            artifact_store,
            execution_cfg=cfg["evaluation"]['cfg'],
        )

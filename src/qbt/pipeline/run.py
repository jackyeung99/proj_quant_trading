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



def run_pipeline(storage, paths, cfg):
    if cfg["ingestion"]["enabled"]:
        ingest_all_sources(storage, paths,
                           ingestion_cfg=cfg["ingestion"],
                           sources_cfg=cfg["sources"])

    if cfg["silver"]["enabled"]:
        canonicalize_all(storage, paths,
                        #  silver_cfg=cfg["silver"],
                         sources_cfg=cfg["sources"])

    if cfg["gold"]["enabled"]:
        build_gold_model_table(storage, paths,
                               gold_cfg=cfg["gold"])
        
    if cfg["signal"]["enabled"]:
        signal(storage, paths,
                signal_cfg=cfg["signal"])
        
    if cfg["execution"]["enabled"]:
        execute_weights(storage, paths,
                        execution_cfg=cfg["execution"])



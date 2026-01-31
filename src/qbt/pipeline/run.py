from __future__ import annotations

from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths

from qbt.pipeline.ingestion import ingest_all_sources
from qbt.pipeline.silver import canonicalize_all
from qbt.pipeline.gold import build_gold_model_table


def run_pipeline(storage: Storage, paths: StoragePaths, cfg: dict) -> None:

    if cfg["ingestion"]["enabled"]:
        ingest_all_sources(storage, paths, cfg)

    if cfg["silver"]["enabled"]:
        canonicalize_all(storage, paths, cfg)

    if cfg["gold"]["enabled"]:
        build_gold_model_table(storage, paths, cfg)
from __future__ import annotations
import os
import logging
import argparse
from dotenv import load_dotenv
from pathlib import Path

from qbt.storage.storage import make_storage
from qbt.storage.artifacts import LiveStore
from qbt.storage.paths import StoragePaths

from qbt.core.logging import setup_logging
from qbt.utils.config_parser import load_controlled_cfg
from qbt.utils.stamping import new_run_id

from qbt.pipeline.ingestion import ingest_all_sources
from qbt.pipeline.silver import canonicalize_all
from qbt.pipeline.gold import build_gold_model_table

def main():
    load_dotenv()

    # ---- CLI ----
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/run.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # ---- Load base config ----
    cfg = load_controlled_cfg(Path(args.cfg)) 

    cfg = {k : v for k, v in cfg.items() if k in ['storage', 'ingestion', 'sources', 'gold']}

    # ---- HARD-CODE experiment change in sources ----
    # # Example: override start date for this run
    cfg["ingestion"]["start_override"] = "2024-01-01"
    cfg['ingestion']['end_override'] = "2026-01-01"
    cfg['sources']['macro_fred']['enabled'] = True

    cfg['gold']['cfg']['tag'] = 'experiment'
    cfg['gold']['cfg']['daily_assets'] =  ['OVXCLS', 'DCOILWTICO', 'DHHNGSP', 'GASREGCOVW']


    # ---- Per-run id ----
    cfg["run_id"] = new_run_id()

    # ---- Logging ----
    setup_logging(
        level=logging.INFO,
        log_file=None,
        force=True,
    )

    # # ---- Inject API keys ----
    cfg["sources"]["equities_intraday_15m"]["api_key"] = os.getenv("ALPACA_API_KEY")
    cfg["sources"]["equities_intraday_15m"]["api_secret"] = os.getenv("ALPACA_API_SECRET")

    # # ---- Storage ----
    storage = make_storage(cfg)
    paths = StoragePaths()
    artifact_store = LiveStore(storage, paths)

    # # ---- Run ingestion + silver ----
    # ingest_all_sources(storage=storage, paths=paths, ingestion_cfg=cfg['ingestion'], sources_cfg=cfg['sources'])

    # canonicalize_all(
    #     storage=storage,
    #     paths=paths,
    #     sources_cfg=cfg["sources"],
    # )

    build_gold_model_table(
        storage,
        paths,
        gold_cfg=cfg["gold"]['cfg'],
        )

    


if __name__ == "__main__":
    main()
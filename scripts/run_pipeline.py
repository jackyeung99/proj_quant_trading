from __future__ import annotations
import os
import logging
import argparse
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import uuid

from qbt.storage.storage import make_storage
from qbt.storage.artifacts import LiveStore
from qbt.storage.paths import StoragePaths
from qbt.pipeline.run import run_pipeline
from qbt.core.logging import setup_logging
from qbt.utils.config_parser import load_controlled_cfg


def new_run_id() -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run QBT pipeline")
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/run.yaml",
        help="Path to config YAML file",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    cfg = load_controlled_cfg(Path(args.cfg))
    
    # --- per-run id + log path ---
    run_id = new_run_id()
    cfg["run_id"] = run_id
    storage_cfg = cfg["storage"]
    backend = storage_cfg.get("backend", "local")


    if backend == "local":
        base_dir = Path(storage_cfg.get("base_dir", "."))
        log_path = (
            base_dir
            / "artifacts"
            / "runs"
            / f"run_id={run_id}"
            / "logs"
            / "app.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)

        setup_logging(
            level=logging.INFO,
            log_file=log_path,
            force=True,
        )
    else:
        # Cloud / ECS → log to stdout only
        setup_logging(
            level=logging.INFO,
            log_file=None,
            force=True,
        )

    # inject api keys
    cfg["sources"]["equities_intraday_15m"]["api_key"] = os.getenv("ALPACA_API_KEY")
    cfg["sources"]["equities_intraday_15m"]["api_secret"] = os.getenv("ALPACA_API_SECRET")

    storage = make_storage(cfg)
    paths = StoragePaths()
    artifact_store = LiveStore(storage, paths)

    run_pipeline(storage=storage, paths=paths, cfg=cfg, artifact_store=artifact_store)


if __name__ == "__main__":
    main()

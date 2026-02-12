from __future__ import annotations
import os
import logging
from dotenv import load_dotenv
from pathlib import Path

from datetime import datetime
import uuid

from qbt.storage.storage import LocalStorage
from qbt.storage.artifacts import LiveStore
from qbt.storage.paths import StoragePaths
from qbt.pipeline.run import run_pipeline
from qbt.core.logging import setup_logging
from qbt.utils.config_parser import load_controlled_cfg

def new_run_id() -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"

def main():
    load_dotenv()

    cfg = load_controlled_cfg(Path("configs/run.yaml"))

    storage_cfg = cfg["storage"]
    base_dir = Path(storage_cfg.get("base_dir", "."))

    # --- per-run id + log path ---
    run_id = new_run_id()
    cfg["run_id"] = run_id  # make it available everywhere

    log_path = base_dir / "artifacts" / "runs" / f"run_id={run_id}" / "logs" / "app.log"

    setup_logging(
        level=logging.INFO,
        log_file=log_path,
        force=True,
    )

    # inject api keys
    cfg["sources"]["equities_intraday_15m"]["api_key"] = os.getenv("ALPACA_API_KEY")
    cfg["sources"]["equities_intraday_15m"]["api_secret"] = os.getenv("ALPACA_API_SECRET")

    storage = LocalStorage(base_dir=base_dir)
    paths = StoragePaths()
    artifact_store = LiveStore(storage, paths)

    run_pipeline(storage=storage, paths=paths, cfg=cfg, artifact_store=artifact_store)


if __name__ == "__main__":
    main()

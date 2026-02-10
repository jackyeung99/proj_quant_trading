import os
from dotenv import load_dotenv
from pathlib import Path
import yaml

from qbt.storage.storage import LocalStorage
from qbt.storage.paths import StoragePaths
from qbt.pipeline.run import run_pipeline
from qbt.core.logging import setup_logging

from qbt.utils.config_parser import load_controlled_cfg


def main():
    load_dotenv()  # local only; in cloud env vars are already set

    cfg = load_controlled_cfg(Path("configs/run.yaml"))
    
    # inject api keys
    cfg["sources"]["equities_intraday_15m"]["api_key"] = os.getenv("ALPACA_API_KEY")
    cfg["sources"]["equities_intraday_15m"]["api_secret"] = os.getenv("ALPACA_API_SECRET")

    storage_cfg = cfg["storage"]
    storage = LocalStorage(base_dir=Path(storage_cfg.get("base_dir", ".")))
    paths = StoragePaths()



    run_pipeline(storage=storage, paths=paths, cfg=cfg)


if __name__ == "__main__":
    setup_logging()
    main()

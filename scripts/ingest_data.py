import os
from dotenv import load_dotenv
from pathlib import Path
import yaml

from qbt.storage.storage import LocalStorage
from qbt.data.ingestion import ingest
from qbt.core.logging import setup_logging


def main():
    load_dotenv()  # local only; in cloud env vars are already set

    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())

    cfg["sources"]["equities_intraday_15m"]["api_key"] = os.getenv("ALPACA_API_KEY")
    cfg["sources"]["equities_intraday_15m"]["api_secret"] = os.getenv("ALPACA_API_SECRET")

    storage_cfg = cfg["storage"]
    storage = LocalStorage(base_dir=Path(storage_cfg.get("base_dir", ".")))

    ingest(storage=storage, cfg=cfg)


if __name__ == "__main__":
    setup_logging()
    main()

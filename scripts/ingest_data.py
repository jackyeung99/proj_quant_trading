import yaml
from pathlib import Path

import pandas as pd

from qbt.storage.storage import LocalStorage
from qbt.data.sources.yfin import fetch, standardize, validate
from qbt.data import transforms
from qbt.core.logging import setup_logging


def main():
    # ------------------
    # Load config
    # ------------------
    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())

    ingest_cfg = cfg["ingestion"]
    path_cfg = cfg["paths"]
    storage_cfg = cfg["storage"]

    storage = LocalStorage(base_dir=Path(storage_cfg.get("base_dir", ".")))

    symbols = ingest_cfg["symbols"]
    start = ingest_cfg["start"]
    end = ingest_cfg["end"]
    interval = ingest_cfg.get("interval", "1d")
    auto_adjust = ingest_cfg.get("auto_adjust", True)

    # ------------------
    # Fetch raw data
    # ------------------
    df = fetch(
        tickers=symbols,
        start=start,
        end=end,
        interval=interval,
        # auto_adjust=auto_adjust,
    )

    df = standardize(df)
    validate(df)

    # ------------------
    # Write bronze prices
    # ------------------
    df = pd.DataFrame()
    bronze_path = Path(path_cfg["bronze_prefix"]) / "prices_daily.parquet"

    storage.write_parquet(df, bronze_path)

    # ------------------
    # Compute log returns
    # ------------------
    returns = pd.DataFrame(index=df.index)

    for col in df.columns:
        returns[f"{col}_ret"] = transforms.log_returns(df[col])

    returns = returns.dropna()

    # ------------------
    # Write processed returns
    # ------------------
    processed_path = Path(path_cfg["processed_returns_key"])
    storage.write_parquet(returns, processed_path)


if __name__ == "__main__":
    setup_logging()
    main()

# tests/test_sources.py
from __future__ import annotations

import os
import pandas as pd
import pytest
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(".env"))

from qbt.data.sources.source_registry import create_source, available_sources


START = pd.Timestamp("2024-01-02", tz="UTC")
END = pd.Timestamp("2024-01-10", tz="UTC")


def _run_source(provider: str, ticker: str, cfg: dict) -> pd.DataFrame:
    src = create_source(provider, cfg=cfg) 
    df = src.fetch(ticker=ticker, start=START, end=END)
    print(df)
    df = src.standardize(df)
    print(df)
    src.validate(df)
    return df


def test_registry_has_expected_sources():
    names = set(available_sources())
    assert "yfinance" in names


def test_yfinance_daily_smoke():
    source_cfg = {
        "interval": "1d",
        # symbols not needed for a single-ticker source, but harmless if you keep it
        "symbols": ["XLE"],
        # optional knobs if you want deterministic tests
        "chunk_size_days": 120,
        "retries": 2,
        "backoff": 0.5,
    }

    df = _run_source("yfinance", "XLE", source_cfg)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"timestamp", "ticker", "close"}.issubset(df.columns)

    assert df["ticker"].nunique() == 1
    assert (df["ticker"] == "XLE").all()
    assert isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype)

    df2 = df.sort_values(["ticker", "timestamp"])
    assert df2.duplicated(["ticker", "timestamp"]).sum() == 0
    assert df2["timestamp"].is_monotonic_increasing


@pytest.mark.skipif(
    not (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET")),
    reason="ALPACA_API_KEY/ALPACA_API_SECRET not set",
)
def test_alpaca_15m_smoke():

    provider_kwargs = {
        "api_key": os.getenv("ALPACA_API_KEY"),
        "api_secret": os.getenv("ALPACA_API_SECRET"),
        "interval": "15Min",
        "feed": "iex",
        "adjustment": "all",
        # optional tuning for tests
        "limit": 10000,
        "sleep_s": 0.0,   # don’t sleep in tests unless you must
    }

    df = _run_source("alpaca", "XLE", provider_kwargs)

    required = {"timestamp", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}; got={list(df.columns)}"

    assert not df.empty
    assert df["ticker"].nunique() == 1
    assert (df["ticker"] == "XLE").all()

    assert isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype)
    assert df["timestamp"].notna().all()

    df2 = df.sort_values(["ticker", "timestamp"])
    assert df2.duplicated(["ticker", "timestamp"]).sum() == 0
    assert df2.groupby("ticker")["timestamp"].apply(lambda x: x.is_monotonic_increasing).all()

    for c in ["open", "high", "low", "close"]:
        assert (df[c].dropna() > 0).all(), f"Non-positive values found in {c}"
    assert (df["volume"].dropna() >= 0).all(), "Negative volume found"


def test_fred_smoke():
    # For FRED, "ticker" is the series id
    source_cfg = {
        "series": ["VIXCLS", "DGS10", "DGS2"],  # not required for fetch(), but ok
    }

    df = _run_source("fred", "VIXCLS", source_cfg)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Adjust these if your FRED canonical schema differs
    assert {"timestamp", "name", "value"}.issubset(df.columns)
    assert (df["name"] == "VIXCLS").any() or df["name"].nunique() >= 1
    assert isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype)

    # sanity: values should be numeric
    assert pd.api.types.is_numeric_dtype(df["value"])

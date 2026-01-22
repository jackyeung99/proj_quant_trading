import os
import time as _time
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from proj.utils.paths import find_project_root
from proj.data.sources.yfin import standardize, validate
from proj.data.storage import make_storage
from proj.data.ingestion_state import get_last_available_date, compute_fetch_window, update_state
from proj.data.sources import yfin, fred, weather
from proj.data.storage import Storage

from proj.data.merge_helpers import merge_and_dedup


    ##  ONE TIME SCRIPT: get high frequency intraday bars


ROOT = find_project_root()  
load_dotenv(ROOT / ".env")

ALPACA_DATA_BASE = "https://data.alpaca.markets"


def _alpaca_headers() -> dict:
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_API_SECRET in .env file"
        )

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }


def fetch_alpaca_bars_15m(
    symbols: list[str],
    start: str,
    end: str,
    feed: str = "iex",           # "iex" = free tier, "sip" = paid
    adjustment: str = "split",
    limit: int = 10000,
    sleep_s: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """
    Pull historical 15-minute OHLCV bars using:
      GET /v2/stocks/bars
    """
    url = f"{ALPACA_DATA_BASE}/v2/stocks/bars"
    headers = _alpaca_headers()

    params = {
        "symbols": ",".join(symbols),
        "timeframe": "15Min",     # native 15-minute bars
        "start": start,           # ISO-8601 UTC
        "end": end,
        "feed": feed,
        "adjustment": adjustment,
        "limit": limit,
    }

    by_symbol = {s: [] for s in symbols}
    page_token = None

    while True:
        if page_token:
            params["page_token"] = page_token
        else:
            params.pop("page_token", None)

        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:300]}")

        js = r.json()
        bars = js.get("bars", {})

        for sym, rows in bars.items():
            by_symbol[sym].extend(rows)

        page_token = js.get("next_page_token")
        if not page_token:
            break

        if sleep_s:
            _time.sleep(sleep_s)

    out = {}
    for sym, rows in by_symbol.items():
        if not rows:
            out[sym] = pd.DataFrame()
            continue

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["t"], utc=True)

        rename = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
        if "vw" in df.columns:
            rename["vw"] = "VWAP"
        if "n" in df.columns:
            rename["n"] = "Trades"

        df = (
            df.rename(columns=rename)
              .set_index("timestamp")
              .sort_index()
        )

        keep = ["open", "high", "low", "close", "volume"]
        if "VWAP" in df.columns:
            keep.append("VWAP")
        if "Trades" in df.columns:
            keep.append("Trades")

        out[sym] = df[keep]

    return out


def fetch(tickers, start, end):
    # Call ONCE with all tickers (best)
    data_dict = fetch_alpaca_bars_15m(
        symbols=tickers,
        start=start,
        end=end,
        feed="iex",
        adjustment="all",
    )

    frames = []
    for sym, df in data_dict.items():
        if df is None or df.empty:
            continue

        tmp = df.copy()
        tmp["ticker"] = sym
        tmp = tmp.reset_index()  # brings timestamp index back as a column named "timestamp"
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "ticker", "open", "high", "low", "close", "volume"])

    out = (
        pd.concat(frames, ignore_index=True)
          .sort_values(["ticker", "timestamp"])
          .drop_duplicates(["ticker", "timestamp"], keep="last")
          .reset_index(drop=True)
    )

    return out


if __name__ == "__main__":

    
    symbols = ["SPY", "XLE"]

    start = "2021-01-01T00:00:00Z"
    # start = "2025-11-01T00:00:00Z"
    end   = "2026-01-01T00:00:00Z"

    data = fetch(symbols, start, end)
    new_df = standardize(data)
    validate(new_df)
    print(new_df)
    
    #merge/write/state
    cfg = {"storage":
            {"backend": "local",
                "base_dir": "data"}
            }

    store_key = "bronze/equities.parquet"
    storage = make_storage(cfg)
    if storage.exists(store_key):
        old_df = storage.read_parquet(store_key)
        merged = merge_and_dedup(old_df, new_df)
    else:
        merged = new_df

    storage.write_parquet(merged, store_key)

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    na_summary = (
        merged.isna()
        .agg(["sum", "mean"])
        .T
        .rename(columns={"sum": "n_missing", "mean": "pct_missing"})
    )

    na_summary["pct_missing"] *= 100
    print(na_summary.sort_values("pct_missing", ascending=False))
    update_state(
        storage,
        store_key,
        merged,
        pull_start=start,
        pull_end=end,
        meta = None
    )




from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Mapping, Any

from qbt.data.sources.source_registry import register_source
from qbt.data.sources.source_base import DataSource

import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()

# If you want to keep env loading in the entrypoint, DO NOT do it in the class.
# Keep the class pure and pass keys in, or read env inside a tiny helper in __main__.

DateLike = Union[str, pd.Timestamp]


# ----------------------------
# Alpaca 15m Bars (tidy, no pivot)
# ----------------------------

@register_source('alpaca')
class AlpacaBarsSource(DataSource):
    """
    Fetch historical 15-minute OHLCV bars from Alpaca Data API.

    Output: TIDY (long) DataFrame:
      columns: [timestamp, ticker, open, high, low, close, volume, (optional VWAP), (optional Trades)]
    No pivoting.

    Notes:
    - feed="iex" is free tier; "sip" is paid.
    - adjustment can be "raw", "split", "dividend", "all" depending on your plan.
    - Alpaca returns paginated results; this class handles next_page_token.
    """



    def __init__(
        self,
        cfg: Mapping[str, Any] | None = None,
        base_url: str = "https://data.alpaca.markets",
        timeout_s: int = 60,
    ):
        super().__init__(cfg=cfg)

        self.base_url = base_url
        self.timeout_s = int(timeout_s)

        # Allow injection; fall back to env (cloud-friendly)
        self.api_key = cfg.get( 
                'api_key',
                os.getenv("ALPACA_API_KEY")
                )
        self.api_secret = cfg.get(
            'api_secret',
             os.getenv("ALPACA_API_SECRET")
             )
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing Alpaca credentials (api_key/api_secret or env vars)")

        # defaults (cfg can override)

        self.feed = str(cfg.get("feed", "iex"))
        self.adjustment = str(cfg.get("adjustment", "all"))
        self.limit = int(cfg.get("limit", 10000))
        self.sleep_s = float(cfg.get("sleep_s", 0.25))
        self.interval = str(cfg.get('interval', '15Min'))


    def fetch(
        self,
        ticker: str,
        start: DateLike,
        end: DateLike,
        ) -> pd.DataFrame:
            if not ticker or not str(ticker).strip():
                return pd.DataFrame(columns=self._base_cols())

            ticker = str(ticker).strip()

            # allow overrides from kwargs
            

            start_ts, end_ts = self._normalize_utc_range(start, end)

            url = f"{self.base_url}/v2/stocks/bars"
            headers = self._headers()

            params = {
                "symbols": ticker,
                "timeframe": self.interval,
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat(),
                "feed": self.feed,
                "adjustment": self.adjustment,
                "limit": self.limit,
            }

            rows: list[dict] = []
            page_token: Optional[str] = None

            while True:
                if page_token:
                    params["page_token"] = page_token
                else:
                    params.pop("page_token", None)

                r = requests.get(url, headers=headers, params=params, timeout=self.timeout_s)
                if r.status_code != 200:
                    raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:300]}")

                js = r.json()
                bars = js.get("bars", {}) or {}

                sym_rows = bars.get(ticker, [])
                rows.extend(sym_rows)

                page_token = js.get("next_page_token")
                if not page_token:
                    break

                if self.sleep_s:
                    _time.sleep(self.sleep_s)

            if not rows:
                return pd.DataFrame(columns=self._base_cols())

            df = pd.DataFrame(rows)

            # Alpaca schema mapping
            df["timestamp"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
            df["ticker"] = ticker

            rename = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "trades",
            }
            df = df.rename(columns=rename)

            keep = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
            if "vwap" in df.columns:
                keep.append("vwap")
            if "trades" in df.columns:
                keep.append("trades")

            df = df[keep]
        
            out = (
                df.dropna(subset=["timestamp"])
                .sort_values(["timestamp"])
                .drop_duplicates(["timestamp"], keep="last")
                .reset_index(drop=True)
            )

            return out

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Canonical tidy schema + dtypes.
        No pivoting, no index requirement.
        """
        if df is None:
            return pd.DataFrame(columns=self._base_cols())

        df = df.copy()

        # Ensure required columns exist (create if missing)
        for c in self._base_cols():
            if c not in df.columns:
                df[c] = pd.NA

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["ticker"] = df["ticker"].astype("string")
        

        # enforce numeric
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # optional fields
        if "vwap" in df.columns:
            df["vwap"] = pd.to_numeric(df["vwap"], errors="coerce")
        if "trades" in df.columns:
            df["trades"] = pd.to_numeric(df["trades"], errors="coerce")

        df = (
            df.dropna(subset=["timestamp", "ticker"])
            .sort_values(["ticker", "timestamp"])
            .drop_duplicates(["ticker", "timestamp"], keep="last")
            .reset_index(drop=True)
        )
        return df

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate tidy OHLCV bars (no pivot assumptions).
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("alpaca: df must be a DataFrame")
        if df.empty:
            raise ValueError("alpaca: DataFrame is empty")

        required = {"timestamp", "ticker", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"alpaca: missing columns: {sorted(missing)}")

        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if ts.isna().any():
            raise ValueError("alpaca: timestamp contains NaT after parsing")

        if df["ticker"].isna().any():
            raise ValueError("alpaca: ticker contains NaNs")

        # monotonic per ticker (after sorting)
        s = df.sort_values(["ticker", "timestamp"])
        if s.groupby("ticker")["timestamp"].apply(lambda x: x.is_monotonic_increasing).eq(False).any():
            raise ValueError("alpaca: timestamps must be non-decreasing within each ticker")

        # numeric checks
        for c in ["open", "high", "low", "close", "volume"]:
            if not pd.api.types.is_numeric_dtype(df[c]):
                raise ValueError(f"alpaca: {c} must be numeric")

        # plausibility checks (allow NaNs but not non-positive where present)
        for c in ["open", "high", "low", "close"]:
            if (df[c].dropna() <= 0).any():
                raise ValueError(f"alpaca: found non-positive values in {c}")

        if (df["volume"].dropna() < 0).any():
            raise ValueError("alpaca: found negative volume")

        # OHLC sanity where all present
        ohlc = df[["open", "high", "low", "close"]].dropna()
        if not ohlc.empty:
            if (ohlc["high"] < ohlc[["open", "close", "low"]].max(axis=1)).any():
                raise ValueError("alpaca: high is less than max(open, close, low) for some rows")
            if (ohlc["low"] > ohlc[["open", "close", "high"]].min(axis=1)).any():
                raise ValueError("alpaca: low is greater than min(open, close, high) for some rows")

    # ----------------------------
    # Internals
    # ----------------------------

    def _headers(self) -> dict:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing Alpaca API key/secret")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    @staticmethod
    def _base_cols() -> list[str]:
        return ["timestamp", "ticker", "open", "high", "low", "close", "volume"]


    @staticmethod
    def _normalize_utc_range(start: DateLike, end: DateLike) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_ts = pd.to_datetime(start, utc=True)
        end_ts = pd.to_datetime(end, utc=True)
        if end_ts == end_ts.normalize():
            end_ts = end_ts + pd.Timedelta(days=1)
        return start_ts, end_ts
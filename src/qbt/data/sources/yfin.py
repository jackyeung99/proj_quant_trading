from __future__ import annotations

import time
from typing import Union

import pandas as pd
import yfinance as yf

from qbt.data.sources.source_registry import register_source
from qbt.data.sources.source_base import DataSource

DateLike = Union[str, pd.Timestamp]


@register_source("yfinance")
class YFinanceBarsSource(DataSource):
    """
    Single-ticker yfinance fetcher.

    fetch(): returns concatenated raw chunks
    standardize(): enforces canonical tidy schema:
        timestamp, ticker, open, high, low, close, volume
        plus optional extras when present
    """

    def __init__(self) -> None:
        pass

    def fetch(
        self,
        ticker: str,
        start: DateLike,
        end: DateLike,
        *,
        interval: str = "1d",
        chunk_size_days: int = 120,
        retries: int = 3,
        backoff: float = 1.5,
        auto_adjust: bool = True,
        actions: bool = False,
    ) -> pd.DataFrame:
        ticker = str(ticker).strip()
        if not ticker:
            return pd.DataFrame()

        start_ts, end_ts = self._normalize_utc_range(start, end)
        chunks = self.chunk_range(start_ts, end_ts, chunk_days=int(chunk_size_days))

        pieces: list[pd.DataFrame] = []
        for cstart, cend in chunks:
            raw_piece = self._fetch_one_chunk(
                ticker=ticker,
                cstart=cstart,
                cend=cend,
                interval=str(interval),
                retries=int(retries),
                backoff=float(backoff),
                auto_adjust=bool(auto_adjust),
                actions=bool(actions),
            )
            if raw_piece is None or raw_piece.empty:
                continue
            pieces.append(raw_piece)

        if not pieces:
            return pd.DataFrame()

        return pd.concat(pieces, ignore_index=True)

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=self._base_cols())

        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]

        ts_col = None
        for cand in ("timestamp", "Datetime", "Date", "index"):
            if cand in df.columns:
                ts_col = cand
                break
        if ts_col is None:
            ts_col = df.columns[0]

        if ts_col != "timestamp":
            df = df.rename(columns={ts_col: "timestamp"})

        if "ticker" not in df.columns:
            raise ValueError("yfinance: missing 'ticker' column (expected added in fetch)")

        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        for c in self._base_cols():
            if c not in df.columns:
                df[c] = pd.NA

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["ticker"] = df["ticker"].astype(str)

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        for c in ["adj_close", "dividends", "stock_splits"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = (
            df.dropna(subset=["timestamp", "ticker"])
              .sort_values(["ticker", "timestamp"])
              .drop_duplicates(["ticker", "timestamp"], keep="last")
              .reset_index(drop=True)
        )

        base = self._base_cols()
        extras = [c for c in ["adj_close", "dividends", "stock_splits"] if c in df.columns]
        return df[base + extras]

    def validate(self, df: pd.DataFrame) -> None:
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("yfinance: df must be a DataFrame")
        if df.empty:
            raise ValueError("yfinance: DataFrame is empty")

        required = set(self._base_cols())
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"yfinance: missing columns: {sorted(missing)}")

        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if ts.isna().any():
            raise ValueError("yfinance: timestamp contains NaT after parsing")

        if df["ticker"].nunique() != 1:
            raise ValueError("yfinance: expected a single ticker")

        for c in ["open", "high", "low", "close", "volume"]:
            if not pd.api.types.is_numeric_dtype(df[c]):
                raise ValueError(f"yfinance: {c} must be numeric")

        for c in ["open", "high", "low", "close"]:
            if (df[c].dropna() <= 0).any():
                raise ValueError(f"yfinance: found non-positive values in {c}")
        if (df["volume"].dropna() < 0).any():
            raise ValueError("yfinance: found negative volume")

    def _fetch_one_chunk(
        self,
        *,
        ticker: str,
        cstart: pd.Timestamp,
        cend: pd.Timestamp,
        interval: str,
        retries: int,
        backoff: float,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        for attempt in range(retries):
            try:
                raw = yf.download(
                    tickers=ticker,
                    start=cstart.tz_convert(None),
                    end=cend.tz_convert(None),
                    interval=interval,
                    auto_adjust=auto_adjust,
                    actions=actions,
                    progress=False,
                    threads=False,
                    group_by="column",
                )

                if raw is None or raw.empty:
                    raise RuntimeError(f"Empty dataframe for {ticker}")

                raw = raw.copy()
                out = raw.reset_index()
                out["ticker"] = ticker
                return out

            except Exception:
                time.sleep(backoff * (2**attempt))

        return pd.DataFrame()

    @staticmethod
    def chunk_range(
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        chunk_days: int,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        cur = start
        step = pd.Timedelta(days=chunk_days)
        while cur < end:
            nxt = min(cur + step, end)
            out.append((cur, nxt))
            cur = nxt
        return out

    @staticmethod
    def _normalize_utc_range(
        start: DateLike,
        end: DateLike,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_ts = pd.to_datetime(start, utc=True)
        end_ts = pd.to_datetime(end, utc=True)
        if end_ts == end_ts.normalize():
            end_ts = end_ts + pd.Timedelta(days=1)
        return start_ts, end_ts

    @staticmethod
    def _base_cols() -> list[str]:
        return ["timestamp", "ticker", "open", "high", "low", "close", "volume"]

from __future__ import annotations

from typing import List, Sequence, Union
import numpy as np
import pandas as pd
import yfinance as yf
import time 

DateLike = Union[str, pd.Timestamp]


def _normalize_range(start: DateLike, end: DateLike) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    # treat date-only end as inclusive (end-of-day)
    if end_ts == end_ts.normalize():
        end_ts = end_ts + pd.Timedelta(days=1)

    return start_ts, end_ts


def _chunks(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Half-open intervals: [cstart, cend)
    """
    out = []
    cur = start
    step = pd.Timedelta(days=chunk_days)
    while cur < end:
        nxt = min(cur + step, end)
        out.append((cur, nxt))
        cur = nxt
    return out


def _fetch_one_chunk(
    ticker: str,
    cstart: pd.Timestamp,
    cend: pd.Timestamp,
    interval: str,
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """
    Fetch a single ticker for a single chunk. Returns tidy: timestamp,ticker,close
    Uses yf.download with retries to handle Yahoo index flakiness (e.g. ^VIX).
    """

    last_err = None

    for attempt in range(retries):
        try:
            df = yf.download(
                tickers=ticker,
                start=cstart.tz_convert(None),
                end=cend.tz_convert(None),
                interval=interval,
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
                group_by="column",
            )

            # Yahoo sometimes returns empty df without raising
            if df is None or df.empty:
                raise RuntimeError(f"Empty dataframe for {ticker}")

            df = df.copy()
            df.index = pd.to_datetime(df.index, utc=True)

            # Defensive close column handling
            if "Close" not in df.columns:
                close_col = next(
                    (c for c in df.columns if str(c).lower() == "close"), None
                )
                if close_col is None:
                    raise RuntimeError(f"No Close column for {ticker}")
                df.rename(columns={close_col: "Close"}, inplace=True)

            tidy = (
                df[["Close"]]
                .rename(columns={"Close": "close"})
                .reset_index()
            )
            tidy.columns = ["timestamp", "close"]
            tidy["ticker"] = ticker

            return tidy[["timestamp", "ticker", "close"]]

        except KeyError as e:
            # yfinance Yahoo index failure (KeyError('chart'))
            last_err = e
        except Exception as e:
            last_err = e

        # Exponential backoff
        time.sleep(backoff * (2 ** attempt))

    # Final fallback: return empty tidy frame (consistent with your pipeline)
    return pd.DataFrame(columns=["timestamp", "ticker", "close"])


def _repair_split_like_jumps(prices: pd.Series, max_abs_logret: float = 0.4) -> pd.Series:
    """
    Repairs split/adjustment glitches by rescaling the *post-jump* segment.
    For ETFs, a single-bar +/-40% is almost surely a data issue.
    """
    s = prices.dropna().copy()
    if s.empty or len(s) < 3:
        return prices

    lr = np.log(s).diff()
    bad = lr.abs() > max_abs_logret
    if not bad.any():
        return prices

    fixed = prices.copy()
    bad_times = lr.index[bad]

    for ts in bad_times:
        loc = s.index.get_loc(ts)
        if loc == 0:
            continue
        prev_ts = s.index[loc - 1]
        factor = s.loc[prev_ts] / s.loc[ts]
        fixed.loc[fixed.index >= ts] = fixed.loc[fixed.index >= ts] * factor

    return fixed


def fetch(
    tickers: Union[str, Sequence[str]],
    start: DateLike,
    end: DateLike,
    interval: str = "60m",
    chunk_size_days: int = 30,
    repair_jumps: bool = True,
    max_abs_logret: float = 0.4,
) -> pd.DataFrame:
    """
    Robust multi-ticker intraday fetch (looped per ticker + chunked).

    Returns tidy DataFrame with columns: [timestamp, ticker, close].
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.replace(",", " ").split() if t.strip()]
    tickers = list(tickers)

    start_ts, end_ts = _normalize_range(start, end)
    chunks = _chunks(start_ts, end_ts, chunk_days=chunk_size_days)

    pieces: List[pd.DataFrame] = []

    for ticker in tickers:
        t_pieces: List[pd.DataFrame] = []

        for cstart, cend in chunks:
            tidy = _fetch_one_chunk(ticker, cstart, cend, interval=interval)
            if tidy.empty:
                continue

            # enforce half-open trimming [cstart, cend)
            tidy = tidy[(tidy["timestamp"] >= cstart) & (tidy["timestamp"] < cend)]
            t_pieces.append(tidy)

        if not t_pieces:
            continue

        tdf = (
            pd.concat(t_pieces, ignore_index=True)
              .sort_values("timestamp")
              .drop_duplicates(["timestamp"], keep="last")
        )

        # optional repair for split-like glitches
        if repair_jumps:
            s = tdf.set_index("timestamp")["close"]
            s_fixed = _repair_split_like_jumps(s, max_abs_logret=max_abs_logret)
            tdf = s_fixed.reset_index()
            tdf["ticker"] = ticker
            tdf = tdf[["timestamp", "ticker", "close"]]

        pieces.append(tdf)

    if not pieces:
        return pd.DataFrame(columns=["timestamp", "ticker", "close"])

    out = (
        pd.concat(pieces, ignore_index=True)
          .sort_values(["ticker", "timestamp"])
          .drop_duplicates(["ticker", "timestamp"], keep="last")
          .reset_index(drop=True)
    )

    return out



def standardize(df: pd.DataFrame) -> pd.DataFrame:
    

    df = df.copy()

    # Ensure timestamp is datetime + UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    wide = (
        df.pivot_table(
            index="timestamp",
            columns="ticker",
            values="close",
            aggfunc="last",   # safe if duplicates exist
        )
        .sort_index()
    )

    return wide

def validate(df: pd.DataFrame) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("yfinance: df must be a DataFrame")
    if df.empty:
        raise ValueError("yfinance: DataFrame is empty")

    # Index checks
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("yfinance: index must be a DatetimeIndex (timestamp)")
    if df.index.hasnans:
        raise ValueError("yfinance: index contains NaNs")
    if df.index.duplicated().any():
        raise ValueError("yfinance: duplicate timestamps in index")
    if not df.index.is_monotonic_increasing:
        raise ValueError("yfinance: timestamps must be sorted increasing")

    # Column checks
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("yfinance: columns should not be MultiIndex after pivot")
    if df.shape[1] == 0:
        raise ValueError("yfinance: no ticker columns found")
    if df.columns.duplicated().any():
        raise ValueError("yfinance: duplicate ticker columns found")

    # Data checks
    if df.isna().all().any():
        bad = df.columns[df.isna().all()].tolist()
        raise ValueError(f"yfinance: tickers entirely NaN: {bad}")

    # Ensure numeric (close should be numeric)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"yfinance: non-numeric ticker columns: {non_numeric}")

    # Optional plausibility: prices should be > 0 whenever present
    if (df.dropna(how="all").le(0)).any().any():
        raise ValueError("yfinance: found non-positive prices")

if __name__ == "__main__":

    features = ["SPY", "XLE"]    
    start = "2025-12-12"
    end = "2025-12-20"

    df = fetch(features, start, end)
    df = standardize(df)

    validate(df)
    print(df.head())

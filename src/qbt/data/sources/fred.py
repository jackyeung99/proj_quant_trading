from __future__ import annotations

from typing import Any, Mapping
import pandas as pd
from pandas_datareader import data as pdr

from qbt.data.sources.source_registry import register_source
from qbt.data.sources.source_base import DataSource


@register_source("fred")
class FredMacroSource(DataSource):
    """
    Single-series FRED fetcher (tidy, no pivot).

    Output columns:
      [timestamp, name, value]
    """

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        super().__init__(cfg=cfg)
        self.interval =  str(cfg.get("interval", "1d"))

    def fetch(self, ticker: str, start, end) -> pd.DataFrame:
        ticker = str(ticker).strip()
        if not ticker:
            return pd.DataFrame(columns=["timestamp", "name", "value"])

        # pdr returns a DataFrame indexed by DATE (tz-naive)
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        if getattr(start, "tzinfo", None) is not None:
            start = start.tz_convert(None)
        if getattr(end, "tzinfo", None) is not None:
            end = end.tz_convert(None)

        raw = pdr.DataReader(ticker, "fred", start=start, end=end)
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["timestamp", "name", "value"])

        out = raw.reset_index()

        # timestamp column can be "DATE" or "Date" depending on versions
        ts_col = None
        for cand in ("DATE", "Date", "date", "index"):
            if cand in out.columns:
                ts_col = cand
                break
        if ts_col is None:
            ts_col = out.columns[0]

        out = out.rename(columns={ts_col: "timestamp"})

        # value column is the ticker name
        if ticker not in out.columns:
            # sometimes DataReader returns a single unnamed column; handle defensively
            val_col = next((c for c in out.columns if c not in ("timestamp",)), None)
            if val_col is None:
                return pd.DataFrame(columns=["timestamp", "name", "value"])
            out = out.rename(columns={val_col: "value"})
        else:
            out = out.rename(columns={ticker: "value"})

        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out["name"] = ticker
        out["value"] = pd.to_numeric(out["value"], errors="coerce")

        out = (
            out
            .set_index("timestamp")
            .sort_index()
            .resample(self.interval)
            .ffill()
            .reset_index()
        )

        return out[["timestamp", "name", "value"]]

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame(columns=["timestamp", "name", "value"])
        df = df.copy()

        for c in ["timestamp", "name", "value"]:
            if c not in df.columns:
                df[c] = pd.NA

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["name"] = df["name"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return (
            df.dropna(subset=["timestamp", "name"])
              .sort_values(["name", "timestamp"])
              .drop_duplicates(["name", "timestamp"], keep="last")
              .reset_index(drop=True)
        )

    def validate(self, df: pd.DataFrame) -> None:
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("fred: df must be a DataFrame")
        if df.empty:
            raise ValueError("fred: DataFrame is empty")

        required = {"timestamp", "name", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"fred: missing columns: {sorted(missing)}")

        if pd.to_datetime(df["timestamp"], utc=True, errors="coerce").isna().any():
            raise ValueError("fred: timestamp contains NaT after parsing")

        if df["name"].isna().any():
            raise ValueError("fred: name contains NaNs")

        if not pd.api.types.is_numeric_dtype(df["value"]):
            raise ValueError("fred: value must be numeric")

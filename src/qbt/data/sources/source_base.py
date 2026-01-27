from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Callable, Optional
import pandas as pd



class DataSource(Protocol):
    """
    A "source adapter" contract: fetch -> standardize -> validate.
    Each provider implements these 3 methods.
    """

    name: str

    def fetch(self, *, start: pd.Timestamp, end: pd.Timestamp, cfg: dict) -> pd.DataFrame: ...
    def standardize(self, df: pd.DataFrame, *, cfg: dict) -> pd.DataFrame: ...
    def validate(self, df: pd.DataFrame, *, cfg: dict) -> None: ...


@dataclass(frozen=True)
class SourceSpec:
    """
    How this specific ingestion task should run for a single dataset (e.g., equities_1m, equities_1d, macro, etc.)
    """
    id: str               # e.g. "equities_intra"
    provider: str         # e.g. "yfinance", "alpaca", "fred"
    store_path: str       # where to write in storage
    enabled: bool = True
    cfg: dict = None      # provider-specific config (symbols, interval, etc.)

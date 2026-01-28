
from __future__ import annotations

import pandas as pd
import numpy as np 



def compute_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    if kind == "log":
        return np.log(prices).diff()
    if kind == "simple":
        return prices.pct_change()
    raise ValueError(f"Unknown return_kind={kind}")

def realized_vol(ret: pd.DataFrame, window: int) -> pd.DataFrame:
    # simple RV proxy at bar frequency: sqrt(sum(r^2))
    return (ret.pow(2).rolling(window).sum()).pow(0.5)

def realized_var(ret: pd.DataFrame, window: int) -> pd.DataFrame:
    # simple RV proxy at bar frequency: sqrt(sum(r^2))
    return ret.pow(2).rolling(window).sum()

def momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    # log momentum
    return np.log(prices).diff(window)
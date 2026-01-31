import pandas as pd
import numpy as np

def intra_rv(df: pd.DataFrame, *, close_col: str = "close") -> float:
    # Realized variance: sum of squared intraday log returns
    c = pd.to_numeric(df[close_col], errors="coerce").astype(float)
    r = np.log(c).diff()
    rv = float(np.nansum((r.values) ** 2))
    return rv

def intra_rvol(df: pd.DataFrame, *, close_col: str = "close") -> float:
    # Realized volatility: sqrt(realized variance)
    return float(np.sqrt(intra_rv(df, close_col=close_col)))

def intra_range(df: pd.DataFrame, *, high_col: str = "high", low_col: str = "low") -> float:
    hi = pd.to_numeric(df[high_col], errors="coerce").astype(float)
    lo = pd.to_numeric(df[low_col], errors="coerce").astype(float)
    return float(np.nanmax(hi.values) - np.nanmin(lo.values))

def intra_bipower_var(df: pd.DataFrame, *, close_col: str = "close") -> float:
    c = pd.to_numeric(df[close_col], errors="coerce").astype(float)
    r = np.log(c).diff().dropna().values
    if len(r) < 2:
        return np.nan
    mu1 = np.sqrt(2 / np.pi)
    bv = (1 / (mu1 ** 2)) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    return float(bv)

def intra_return_skew(df: pd.DataFrame, *, close_col: str = "close") -> float:
    r = np.log(pd.to_numeric(df[close_col], errors="coerce")).diff()
    return float(r.skew())


def intra_return_kurt(df: pd.DataFrame, *, close_col: str = "close") -> float:
    r = np.log(pd.to_numeric(df[close_col], errors="coerce")).diff()
    return float(r.kurtosis())

def intra_jump_var(df: pd.DataFrame, *, close_col: str = "close") -> float:
    rv = intra_rv(df, close_col=close_col)
    bv = intra_bipower_var(df, close_col=close_col)
    if not np.isfinite(rv) or not np.isfinite(bv):
        return np.nan
    return float(max(rv - bv, 0.0))

def intra_nbars(df: pd.DataFrame) -> float:
    return float(len(df))

INTRA_FEATURE_FUNCS = {
    "rv": intra_rv,
    "rvol": intra_rvol,
    "range": intra_range,
    "nbars": intra_nbars,
    "bv": intra_bipower_var,
    "jump_var": intra_jump_var,
    "skew": intra_return_skew,
    "kurt": intra_return_kurt,
}
from __future__ import annotations

import numpy as np
import pandas as pd

from qbt.features.forecasts import ewma_vol_forecast_next, garch_vol_forecast_next

def returns(
    df: pd.DataFrame,
    *,
    col: str = "close",
    kind: str = "log",
    out: str = "ret_1",
) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    if kind == "log":
        df[out] = np.log(s).diff()
    elif kind == "simple":
        df[out] = s.pct_change()
    else:
        raise ValueError(f"returns.kind must be 'log' or 'simple', got {kind}")
    return df


def momentum(
    df: pd.DataFrame,
    *,
    col: str = "close",
    windows: list[int] = [20],
    prefix: str = "mom_",
) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    for w in windows:
        df[f"{prefix}{w}"] = s / s.shift(w) - 1.0
    return df


def rolling_mean(
    df: pd.DataFrame,
    *,
    col: str = "close",
    windows: list[int] = [20],
    prefix: str = "sma_",
    min_periods: int | None = None,
) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    for w in windows:
        mp = w if min_periods is None else min_periods
        df[f"{prefix}{w}"] = s.rolling(w, min_periods=mp).mean()
    return df


def rolling_std(
    df: pd.DataFrame,
    *,
    col: str = "ret_1",
    windows: list[int] = [20],
    prefix: str = "vol_",
    min_periods: int | None = None,
    annualize: bool = False,
    periods_per_year: float = 252.0,
) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    for w in windows:
        mp = w if min_periods is None else min_periods
        v = s.rolling(w, min_periods=mp).std(ddof=1)
        if annualize:
            v = v * np.sqrt(periods_per_year)
        df[f"{prefix}{w}"] = v
    return df


def rolling_sharpe(
    df: pd.DataFrame,
    *,
    ret_col: str = "ret_1",
    windows: list[int] = [63],
    prefix: str = "sharpe_",
    annualize: bool = True,
    periods_per_year: float = 252.0,
    min_periods: int | None = None,
) -> pd.DataFrame:
    r = pd.to_numeric(df[ret_col], errors="coerce").astype(float)
    for w in windows:
        mp = w if min_periods is None else min_periods
        mu = r.rolling(w, min_periods=mp).mean()
        sd = r.rolling(w, min_periods=mp).std(ddof=1)
        sh = mu / sd
        if annualize:
            sh = sh * np.sqrt(periods_per_year)
        df[f"{prefix}{w}"] = sh
    return df


def max_drawdown(
    df: pd.DataFrame,
    *,
    col: str = "close",
    window: int = 252,
    out: str = "mdd_252",
    min_periods: int | None = None,
) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    mp = window if min_periods is None else min_periods
    roll_max = s.rolling(window, min_periods=mp).max()
    dd = s / roll_max - 1.0
    df[out] = dd.rolling(window, min_periods=mp).min()
    return df


def zscore(
    df: pd.DataFrame,
    *,
    col: str = "ret_1",
    window: int = 60,
    out: str = "z_60",
    min_periods: int | None = None,
) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    mp = window if min_periods is None else min_periods
    mu = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std(ddof=1)
    df[out] = (s - mu) / sd
    return df


def trend_slope(
    df: pd.DataFrame,
    *,
    col: str = "close",
    window: int = 60,
    out: str = "trend_slope_60",
    use_log: bool = True,
    min_periods: int | None = None,
) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    y = np.log(s) if use_log else s

    mp = window if min_periods is None else min_periods

    def slope(arr: np.ndarray) -> float:
        if np.any(~np.isfinite(arr)):
            return np.nan
        x = np.arange(len(arr), dtype=float)
        # slope of y ~ a + b*x
        b = np.polyfit(x, arr, 1)[0]
        return float(b)

    df[out] = y.rolling(window, min_periods=mp).apply(lambda a: slope(a.values), raw=False)
    return df


def parkinson_vol(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    out: str = "parkinson_vol",
) -> pd.DataFrame:
    hi = pd.to_numeric(df[high_col], errors="coerce").astype(float)
    lo = pd.to_numeric(df[low_col], errors="coerce").astype(float)
    # Parkinson variance estimator (daily)
    var = (1.0 / (4.0 * np.log(2.0))) * (np.log(hi / lo) ** 2)
    df[out] = np.sqrt(var)
    return df


DAILY_TRANSFORMS = {
    "returns": returns,
    "momentum": momentum,

    "rolling_mean": rolling_mean,
    "rolling_std": rolling_std,
    "rolling_sharpe": rolling_sharpe,

    "max_drawdown": max_drawdown,
    "zscore": zscore,

    "trend_slope": trend_slope,
    "parkinson_vol": parkinson_vol,
    
    "ewma_forecast": ewma_vol_forecast_next,
    "garch_forecast": garch_vol_forecast_next,
}

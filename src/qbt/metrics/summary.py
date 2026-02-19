from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

def _sharpe(ret: pd.Series, ann_factor: int) -> float:
    mu = ret.mean()
    sd = ret.std(ddof=0)
    if sd <= 0:
        return np.nan
    return float((mu / sd) * np.sqrt(ann_factor))


def _cagr(equity: pd.Series, ann_factor: int) -> float:
    if len(equity) < 2:
        return np.nan
    # infer periods per year from ann_factor (assume daily)
    years = (len(equity) - 1) / ann_factor
    if years <= 0:
        return np.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def _annual_vol(ret: pd.Series, ann_factor: int) -> float:
    return float(ret.std(ddof=0) * np.sqrt(ann_factor))



def _max_dd(equity: pd.Series):
    peak = equity.cummax()
    dd = equity / peak - 1.0

    return dd.min()


def compute_metrics_simple(
    ret: pd.Series,
    ann_factor: int = 252
) -> Dict[str, Any]:
    r = ret.dropna().astype(float)
    if len(r) == 0:
        return {"n_obs": 0}

    mean = r.mean()
    vol = r.std(ddof=1)

    sharpe = (mean / vol) * np.sqrt(ann_factor) if vol > 0 else np.nan

    # Correct compounding for simple returns
    equity = (1.0 + r).cumprod()

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = dd.min()

    cagr = equity.iloc[-1] ** (ann_factor / len(r)) - 1.0

    return {
        "n_obs": int(len(r)),
        "mean_daily": float(mean),
        "vol_daily": float(vol),
        "sharpe": float(sharpe) if sharpe == sharpe else None,
        "cagr": float(cagr) if cagr == cagr else None,
        "max_dd": float(max_dd),
    }


def compute_metrics_log(
    log_ret: pd.Series,
    ann_factor: int = 252
) -> Dict[str, Any]:
    r = log_ret.dropna().astype(float)
    if len(r) == 0:
        return {"n_obs": 0}

    mean = r.mean()
    vol = r.std(ddof=1)

    sharpe = (mean / vol) * np.sqrt(ann_factor) if vol > 0 else np.nan

    # Correct compounding for log returns
    equity = np.exp(r.cumsum())

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = dd.min()

    # Annualized return from log returns
    cagr = np.exp(mean * ann_factor) - 1.0

    return {
        "n_obs": int(len(r)),
        "mean_daily_log": float(mean),
        "vol_daily_log": float(vol),
        "sharpe_log": float(sharpe) if sharpe == sharpe else None,
        "cagr": float(cagr),
        "max_dd": float(max_dd),
    }
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_strategy_execution(
    returns: pd.DataFrame,          # [T x N] simple returns, columns=tickers
    weights: pd.DataFrame,          # [T x N] desired weights, columns=tickers
    weight_lag: int = 1,
    costs_bps: float = 0.0,         # optional turnover cost in bps
    normalize: bool = False,        # if True, rescale weights to sum(abs)=1 each day
) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty pandas DataFrame")
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        raise ValueError("weights must be a non-empty pandas DataFrame")

    # --- align on index + columns ---
    idx = returns.index
    tickers = list(returns.columns)

    # Reindex weights to returns' shape; missing -> 0 exposure
    w = (
        weights.reindex(index=idx, columns=tickers)
        .astype(float)
        .fillna(0.0)
    )

    r = returns.astype(float)

    # Optional normalization (helps if strategy outputs raw signals)
    if normalize:
        denom = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.div(denom, axis=0).fillna(0.0)

    # --- timing: lag weights to avoid lookahead ---
    w_lagged = w.shift(weight_lag).fillna(0.0)

    # --- asset-level contributions ---
    asset_ret_gross = w_lagged.mul(r, axis=0)

    # --- simple transaction costs: bps * turnover ---
    if costs_bps and costs_bps > 0:
        turnover = w_lagged.diff().abs().sum(axis=1).fillna(0.0)
        cost = (costs_bps / 1e4) * turnover  # bps -> decimal
    else:
        turnover = pd.Series(0.0, index=idx, dtype=float)
        cost = pd.Series(0.0, index=idx, dtype=float)

    # --- portfolio returns ---
    port_ret_gross = asset_ret_gross.sum(axis=1)
    port_ret_net = port_ret_gross - cost

    equity_gross = (1.0 + port_ret_gross).cumprod()
    equity_net = (1.0 + port_ret_net).cumprod()

    out = pd.DataFrame(
        {
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "equity_gross": equity_gross,
            "equity_net": equity_net,
            "turnover": turnover,
            "cost": cost,
        },
        index=idx,
    )

    # Include lagged weights + asset contributions for debugging/analysis
    out = pd.concat(
        [
            out,
            w_lagged.add_prefix("weight_"),
            asset_ret_gross.add_prefix("asset_ret_gross_"),
        ],
        axis=1,
    )
    return out

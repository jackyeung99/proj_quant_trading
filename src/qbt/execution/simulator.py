from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


def simulate_strategy_execution(
    returns: pd.DataFrame,          # [T x N] asset returns (log or simple)
    weights: pd.DataFrame,          # [T x N] desired weights
    *,
    weight_lag: int = 1,
    transaction_cost_bps: float = 0.0,
    normalize: bool = False,
    asset_return_type: Literal["log", "simple"] = "log",
    add_buy_and_hold: bool = True,
    buy_and_hold_asset: str | None = None,   # default: first column
) -> pd.DataFrame:
    """
    - Converts asset returns to SIMPLE internally (if provided as log)
    - Computes portfolio SIMPLE returns: sum_i w_{t-lag,i} * r_{t,i}^{simple}
    - Applies transaction costs in SIMPLE space: r_net = r_gross - cost
    - Adds buy & hold equity/returns for a chosen asset (or first column)
    """

    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty pandas DataFrame")
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        raise ValueError("weights must be a non-empty pandas DataFrame")
    if asset_return_type not in ("log", "simple"):
        raise ValueError("asset_return_type must be 'log' or 'simple'")
    if weight_lag < 0:
        raise ValueError("weight_lag must be >= 0")

    # --- align on index + columns ---
    idx = returns.index
    tickers = list(returns.columns)

    w = weights.reindex(index=idx, columns=tickers).astype(float).fillna(0.0)
    r_in = returns.reindex(index=idx, columns=tickers).astype(float)

    # --- convert asset returns to SIMPLE if needed ---
    if asset_return_type == "log":
        # simple = exp(log) - 1
        r = np.expm1(r_in)
    else:
        r = r_in

    # --- optional normalization (exposure control) ---
    if normalize:
        denom = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.div(denom, axis=0).fillna(0.0)

    # --- timing: lag weights ---
    w_lagged = w.shift(weight_lag).fillna(0.0)

    # --- portfolio gross returns in SIMPLE space ---
    asset_ret_gross = w_lagged.mul(r, axis=0)
    port_ret_gross = asset_ret_gross.sum(axis=1)

    # --- transaction costs (SIMPLE) ---
    if transaction_cost_bps and transaction_cost_bps > 0:
        turnover = w_lagged.diff().abs().sum(axis=1).fillna(0.0)
        cost = (transaction_cost_bps / 1e4) * turnover
    else:
        turnover = pd.Series(0.0, index=idx, dtype=float)
        cost = pd.Series(0.0, index=idx, dtype=float)

    port_ret_net = port_ret_gross - cost

    # --- equity curves (SIMPLE compounding) ---
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

    # --- buy & hold benchmark (SIMPLE) ---
    if add_buy_and_hold:
        if buy_and_hold_asset is None:
            buy_and_hold_asset = tickers[0]
        if buy_and_hold_asset not in tickers:
            raise ValueError(f"buy_and_hold_asset={buy_and_hold_asset!r} not in returns columns")

        bh_ret = r[buy_and_hold_asset].fillna(0.0)
        out["bh_ret"] = bh_ret
        out["bh_equity"] = (1.0 + bh_ret).cumprod()

        # optional: excess vs buy&hold
        out["excess_ret_net"] = out["port_ret_net"] - out["bh_ret"]
        out["excess_equity_net"] = (1.0 + out["excess_ret_net"]).cumprod()

    # --- debugging columns (optional: these can get huge) ---
    out = pd.concat(
        [
            out,
            w_lagged.add_prefix("weight_"),
            asset_ret_gross.add_prefix("asset_ret_gross_"),
        ],
        axis=1,
    )

    return out
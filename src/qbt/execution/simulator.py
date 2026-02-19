from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


def simulate_strategy_execution(
    returns: pd.DataFrame,          # [T x N] returns (simple OR log)
    weights: pd.DataFrame,          # [T x N] desired weights
    weight_lag: int = 1,
    transaction_cost_bps: float = 0.0,
    normalize: bool = False,
    return_type: Literal["simple", "log"] = "simple",
) -> pd.DataFrame:

    if return_type not in ("simple", "log"):
        raise ValueError("return_type must be 'simple' or 'log'")

    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty pandas DataFrame")
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        raise ValueError("weights must be a non-empty pandas DataFrame")

    # --- align on index + columns ---
    idx = returns.index
    tickers = list(returns.columns)

    w = (
        weights.reindex(index=idx, columns=tickers)
        .astype(float)
        .fillna(0.0)
    )

    r = returns.astype(float)

    # --- optional normalization ---
    if normalize:
        denom = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.div(denom, axis=0).fillna(0.0)

    # --- timing: lag weights ---
    w_lagged = w.shift(weight_lag).fillna(0.0)

    # --- asset-level contributions ---
    asset_ret_gross = w_lagged.mul(r, axis=0)

    # --- transaction costs ---
    if transaction_cost_bps and transaction_cost_bps > 0:
        turnover = w_lagged.diff().abs().sum(axis=1).fillna(0.0)
        cost = (transaction_cost_bps / 1e4) * turnover
    else:
        turnover = pd.Series(0.0, index=idx, dtype=float)
        cost = pd.Series(0.0, index=idx, dtype=float)

    # --- portfolio returns ---
    port_ret_gross = asset_ret_gross.sum(axis=1)

    if return_type == "simple":
        port_ret_net = port_ret_gross - cost

        equity_gross = (1.0 + port_ret_gross).cumprod()
        equity_net = (1.0 + port_ret_net).cumprod()

    else:  # LOG RETURNS
        # cost must be applied multiplicatively → convert to log-space
        cost_log = np.log(1.0 - cost.clip(upper=0.999999))

        port_ret_net = port_ret_gross + cost_log

        equity_gross = np.exp(port_ret_gross.cumsum())
        equity_net = np.exp(port_ret_net.cumsum())

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

    # --- debugging columns ---
    out = pd.concat(
        [
            out,
            w_lagged.add_prefix("weight_"),
            asset_ret_gross.add_prefix("asset_ret_gross_"),
        ],
        axis=1,
    )

    return out

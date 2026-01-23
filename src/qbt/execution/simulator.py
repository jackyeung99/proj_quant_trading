import numpy as np
import pandas as pd
from typing import Union, Optional

WeightsLike = Union[pd.DataFrame, pd.Series]

def simulate_strategy_execution(
    returns: pd.DataFrame,                 # index=date, columns=tickers, values=simple returns
    weights: WeightsLike,                  # DataFrame(date x tickers) OR Series(date) OR Series(tickers)
    weight_lag: int = 1,
    costs_bps: float = 0.0,                # optional simple turnover cost in bps
    normalize: bool = False,               # if True, rescale weights to sum(abs)=1 each day
) -> pd.DataFrame:
    if returns.empty:
        raise ValueError("returns is empty")
    if not isinstance(returns.index, pd.Index):
        raise TypeError("returns must be a pandas DataFrame with an index")

    tickers = list(returns.columns)

    # ---- Coerce weights into a (date x tickers) DataFrame ----
    if isinstance(weights, pd.DataFrame):
        w = weights.reindex(index=returns.index, columns=tickers).astype(float)

    elif isinstance(weights, pd.Series):
        # Case A: Series indexed by tickers => static allocation
        if weights.index.equals(pd.Index(tickers)) or set(weights.index) == set(tickers):
            w = pd.DataFrame(np.nan, index=returns.index, columns=tickers, dtype=float)
            w.loc[:, :] = weights.reindex(tickers).astype(float).to_numpy()

        # Case B: Series indexed by dates => scalar exposure each day (applies equally to all assets)
        else:
            s = weights.reindex(returns.index).astype(float)
            w = pd.DataFrame({t: s for t in tickers})

    else:
        raise TypeError("weights must be a pandas DataFrame or Series")

    # Fill missing weights with 0 (no exposure where unspecified)
    w = w.fillna(0.0)

    # Optional normalization (helps if your strategy outputs raw signals)
    if normalize:
        denom = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.div(denom, axis=0).fillna(0.0)

    # ---- Timing: lag weights to avoid lookahead ----
    w_lagged = w.shift(weight_lag).fillna(0.0)

    # ---- Asset-level PnL contributions ----
    asset_ret_gross = w_lagged * returns.astype(float)

    # ---- Simple transaction costs: bps * turnover ----
    # turnover = sum(abs(delta_w)) across tickers per day
    if costs_bps and costs_bps > 0:
        turnover = w_lagged.diff().abs().sum(axis=1).fillna(0.0)
        cost = (costs_bps / 1e4) * turnover  # bps -> decimal
    else:
        turnover = pd.Series(0.0, index=returns.index)
        cost = pd.Series(0.0, index=returns.index)

    # Portfolio return = sum of contributions across assets minus costs
    port_ret_gross = asset_ret_gross.sum(axis=1)
    port_ret_net = port_ret_gross - cost

    equity_gross = (1.0 + port_ret_gross).cumprod()
    equity_net = (1.0 + port_ret_net).cumprod()

    # ---- Output: tidy but information-rich ----
    out = pd.DataFrame(
        {
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "equity_gross": equity_gross,
            "equity_net": equity_net,
            "turnover": turnover,
            "cost": cost,
        },
        index=returns.index,
    )

    # Add weights + contributions as multiindex columns (keeps it scalable)
    weights_out = w_lagged.add_prefix("weight_")
    asset_out   = asset_ret_gross.add_prefix("asset_ret_gross_")

    out = pd.concat([out, weights_out, asset_out], axis=1)
    return out

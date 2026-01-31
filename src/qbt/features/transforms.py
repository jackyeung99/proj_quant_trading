
from __future__ import annotations

import pandas as pd
import numpy as np 
import statsmodels.api as sm



def compute_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    if kind == "log":
        return np.log(prices).diff()
    if kind == "simple":
        return prices.pct_change()
    raise ValueError(f"Unknown return_kind={kind}")

def rolling_realized_vol(ret: pd.DataFrame, window: int) -> pd.DataFrame:
    # simple RV proxy at bar frequency: sqrt(sum(r^2))
    return (ret.pow(2).rolling(window).sum()).pow(0.5)

def rolling_realized_var(ret: pd.DataFrame, window: int) -> pd.DataFrame:
    # simple RV proxy at bar frequency: sqrt(sum(r^2))
    return ret.pow(2).rolling(window).sum()

def momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    # log momentum
    return np.log(prices).diff(window)


def estimate_idiosyncratic_multiindex(
    daily: pd.DataFrame,
    *,
    target: str = "XLE",
    benchmark: str = "SPY",
    window: int = 260,
    return_kind: str = "log",
    close_field: str = "close",
) -> pd.DataFrame:
    """
    Compute rolling CAPM-style idiosyncratic residuals BEFORE flattening.

    Parameters
    ----------
    daily : DataFrame
        WIDE daily table with MultiIndex columns (asset, field).
        Must contain (target, close_field) and (benchmark, close_field).
    target, benchmark : str
        Asset tickers/names as they appear in the column level 0.
    window : int
        Rolling regression window length.
    return_kind : {"log","simple"}
        Return type used for r_t series.
    close_field : str
        Field name for close prices in level 1.

    Returns
    -------
    out : DataFrame
        Same as input but with two additional columns:
          (target, f"beta_{benchmark}")
          (target, f"idio_{benchmark}")
    """
    if not isinstance(daily.columns, pd.MultiIndex):
        raise ValueError("daily must have MultiIndex columns: (asset, field).")

    needed = {(target, close_field), (benchmark, close_field)}
    missing = [c for c in needed if c not in daily.columns]
    if missing:
        raise ValueError(f"Missing required columns for idio calc: {missing}")

    out = daily.copy()

    # --- compute returns from close prices (aligned, wide Series) ---
    p_t = out[(target, close_field)].astype(float)
    p_b = out[(benchmark, close_field)].astype(float)

    if return_kind == "log":
        r_t = np.log(p_t).diff()
        r_b = np.log(p_b).diff()
    elif return_kind == "simple":
        r_t = p_t.pct_change()
        r_b = p_b.pct_change()
    else:
        raise ValueError("return_kind must be 'log' or 'simple'.")

    # Prepare output series
    beta = pd.Series(np.nan, index=out.index, name=(target, f"beta_{benchmark}"))
    idio = pd.Series(np.nan, index=out.index, name=(target, f"idio_{benchmark}"))

    # --- rolling regression ---
    # We need enough rows and finite values
    for i in range(window, len(out.index)):
        yt = r_t.iloc[i - window:i]
        xb = r_b.iloc[i - window:i]

        # drop NaNs inside window
        w = pd.concat([yt, xb], axis=1).dropna()
        if len(w) < max(20, window // 5):  # minimal robustness threshold
            continue

        y = w.iloc[:, 0]
        x = w.iloc[:, 1]
        X = sm.add_constant(x, has_constant="add")

        res = sm.OLS(y, X).fit()

        alpha = float(res.params.get("const", res.params.iloc[0]))
        beta_i = float(res.params.iloc[1])

        beta.iloc[i] = beta_i

        # idio at time i uses current returns (must be finite)
        rt_i = r_t.iloc[i]
        rb_i = r_b.iloc[i]
        if np.isfinite(rt_i) and np.isfinite(rb_i):
            idio.iloc[i] = float(rt_i - (alpha + beta_i * rb_i))

    # attach to out as MultiIndex columns
    out[(target, f"beta_{benchmark}")] = beta.values
    out[(target, f"idio_{benchmark}")] = idio.values

    # keep column order tidy
    out = out.sort_index(axis=1, level=[0, 1])

    return out
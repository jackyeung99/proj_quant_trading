from __future__ import annotations

import numpy as np
import pandas as pd

from arch import arch_model

def ewma_vol_forecast_next(
    df: pd.DataFrame,
    *,
    ret_col: str = "ret_cc",
    lam: float = 0.94,
    out_col: str = "ewma_sigma_ann_next",
    annualize: bool = True,
    periods_per_year: float = 252.0,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    Causal EWMA volatility forecast.

    At timestamp t:
      - uses returns up to and including r[t]
      - EWMA variance computed at t is the forecast for t+1
      - stored at index t

    This is SAFE to persist in gold and use with walk-forward backtests.

    Parameters
    ----------
    ret_col : str
        Column containing returns (daily).
    lam : float
        EWMA decay parameter (RiskMetrics default = 0.94).
    out_col : str
        Output column name.
    annualize : bool
        Whether to annualize volatility.
    periods_per_year : float
        Annualization factor (252 for daily).
    min_obs : int
        Minimum observations before producing output.
    """
    x = df.copy()

    r = pd.to_numeric(x[ret_col], errors="coerce").astype(float)

    # EWMA variance recursion
    var = pd.Series(index=r.index, dtype=float)

    for i, t in enumerate(r.index):
        if i == 0 or not np.isfinite(r.iloc[i]):
            continue

        if i == 1:
            # initialize with first squared return
            var.iloc[i] = r.iloc[i] ** 2
            continue

        var.iloc[i] = lam * var.iloc[i - 1] + (1.0 - lam) * (r.iloc[i] ** 2)

    vol = np.sqrt(var)

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    # enforce minimum history
    vol.iloc[:min_obs] = np.nan

    x[out_col] = vol
    return x


def garch_vol_forecast_next(
    df: pd.DataFrame,
    *,
    ret_col: str = "ret_cc",
    out_col: str = "garch_sigma_ann_next",
    p: int = 1,
    q: int = 1,
    window: int = 750,
    refit_every: int = 5,
    annualize: bool = True,
    periods_per_year: float = 252.0,
    scale: float = 100.0,
) -> pd.DataFrame:
    """
    Adds a one-step-ahead GARCH volatility forecast (next period) as a column.

    Causality:
      - At timestamp t, we use returns up to t to forecast sigma_{t+1}.
      - We store that forecast at timestamp t (for your strategy to produce weights at t,
        which are then shifted to trade at t+1).

    Performance:
      - Fits on rolling window of length `window` (or fewer if early).
      - Refits every `refit_every` steps (otherwise reuses last fit as approximation).
    """
    x = df.copy()

    # compute returns (log or simple—pick one and be consistent with rest of pipeline)
    r = pd.to_numeric(x[ret_col], errors="coerce").astype(float)

    # arch likes percent-ish scaling for numerical stability
    r_scaled = (r * scale).dropna()
    out = pd.Series(index=x.index, dtype=float, name=out_col)

    if len(r_scaled) < max(window // 5, 50):
        x[out_col] = out
        return x

    last_res = None
    last_fit_end = None

    idx = r_scaled.index
    for i in range(len(idx)):
        t = idx[i]

        # rolling window ending at t (inclusive)
        start_i = max(0, i - window + 1)
        r_win = r_scaled.iloc[start_i : i + 1]

        if len(r_win) < 50:
            continue

        # refit periodically
        if (last_res is None) or (last_fit_end is None) or ((i - last_fit_end) >= refit_every):
            am = arch_model(
                r_win,
                mean="Zero",
                vol="GARCH",
                p=p,
                q=q,
                dist="normal",
                rescale=False,
            )
            last_res = am.fit(disp="off")
            last_fit_end = i

        # one-step-ahead forecast from model fit up to t
        f = last_res.forecast(horizon=1, reindex=False)
        var_next_scaled = float(f.variance.values[-1, 0])  # variance of scaled returns

        # convert back: if r_scaled = r * scale, then var(r) = var_scaled / scale^2
        vol = var_next_scaled / (scale ** 2)

        # annualize vol if you want; otherwise store daily sigma

        if annualize:
            vol = float(np.sqrt(max(vol, 0.0)) * np.sqrt(periods_per_year))

        # IMPORTANT: store forecast for t+1 at index t
        out.loc[t] = vol

    x[out_col] = out
    return x

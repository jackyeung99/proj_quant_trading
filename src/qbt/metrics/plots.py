# qbt/metrics/plots.py
from __future__ import annotations

from typing import Optional, Sequence, Dict, Any, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# Helpers
# -----------------------------
def _ensure_dt_index(ts: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts = ts.copy()
        ts.index = pd.to_datetime(ts.index, errors="coerce")
    return ts.sort_index()

def _title(base: str, run_id: Optional[str]) -> str:
    return f"{base}: {run_id}" if run_id else base

def _pick_first_col(ts: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in ts.columns:
            return c
    return None

def _equity_from_returns(ret: pd.Series) -> pd.Series:
    ret = ret.fillna(0.0).astype(float)
    return (1.0 + ret).cumprod()

def _returns_from_equity(eq: pd.Series) -> pd.Series:
    eq = eq.astype(float)
    return eq.pct_change()

def _drawdown(eq: pd.Series) -> pd.Series:
    eq = eq.astype(float)
    peak = eq.cummax()
    return (eq / peak) - 1.0

def _rolling_sharpe(ret: pd.Series, window: int, ann_factor: float = 252.0) -> pd.Series:
    r = ret.astype(float)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=0)
    return np.sqrt(ann_factor) * (mu / sd.replace(0.0, np.nan))

def _rolling_vol(ret: pd.Series, window: int, ann_factor: float = 252.0) -> pd.Series:
    r = ret.astype(float)
    return np.sqrt(ann_factor) * r.rolling(window).std(ddof=0)

def _common_layout(fig, title: str):
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        title_x=0.02,
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)
    return fig


# -----------------------------
# 1) Equity curve
# -----------------------------
def equity_curve_fig(
    ts: pd.DataFrame,
    run_id: Optional[str] = None,
    meta: Optional[dict] = None,
    equity_col: Optional[str] = None,
    ret_col_candidates: Sequence[str] = ("ret_net", "ret_gross", "ret"),
) -> go.Figure:
    ts = _ensure_dt_index(ts)

    eq_col = equity_col or _pick_first_col(ts, ("equity_net", "equity_gross"))
    if eq_col is not None:
        y = ts[eq_col].astype(float)
        fig = px.line(y, title=_title("Equity", run_id))
        return _common_layout(fig, _title("Equity", run_id))

    rc = _pick_first_col(ts, ret_col_candidates)
    if rc is None:
        fig = px.line(title=_title("Equity (missing equity/returns columns)", run_id))
        return _common_layout(fig, _title("Equity", run_id))

    eq = _equity_from_returns(ts[rc])
    fig = px.line(eq, title=_title("Equity (from returns)", run_id))
    return _common_layout(fig, _title("Equity", run_id))


# -----------------------------
# 2) Drawdown
# -----------------------------
def drawdown_fig(
    ts: pd.DataFrame,
    run_id: Optional[str] = None,
    meta: Optional[dict] = None,
    equity_col: Optional[str] = None,
    ret_col_candidates: Sequence[str] = ("ret_net", "ret_gross", "ret"),
) -> go.Figure:
    ts = _ensure_dt_index(ts)

    eq_col = equity_col or _pick_first_col(ts, ("equity_net", "equity_gross"))
    if eq_col is not None:
        eq = ts[eq_col].astype(float)
    else:
        rc = _pick_first_col(ts, ret_col_candidates)
        if rc is None:
            fig = px.line(title=_title("Drawdown (missing equity/returns columns)", run_id))
            return _common_layout(fig, _title("Drawdown", run_id))
        eq = _equity_from_returns(ts[rc])

    dd = _drawdown(eq)
    fig = px.line(dd, title=_title("Drawdown", run_id))
    fig.update_yaxes(tickformat=".1%")
    return _common_layout(fig, _title("Drawdown", run_id))


# -----------------------------
# 3) Returns time series
# -----------------------------
def returns_fig(
    ts: pd.DataFrame,
    run_id: Optional[str] = None,
    meta: Optional[dict] = None,
    ret_col: Optional[str] = None,
) -> go.Figure:
    ts = _ensure_dt_index(ts)

    rc = ret_col or _pick_first_col(ts, ("ret_net", "ret_gross", "ret"))
    if rc is None:
        # try compute from equity
        eqc = _pick_first_col(ts, ("equity_net", "equity_gross"))
        if eqc is None:
            fig = px.line(title=_title("Returns (missing ret/equity columns)", run_id))
            return _common_layout(fig, _title("Returns", run_id))
        r = _returns_from_equity(ts[eqc]).dropna()
        fig = px.line(r, title=_title("Returns (from equity)", run_id))
        return _common_layout(fig, _title("Returns", run_id))

    r = ts[rc].astype(float)
    fig = px.line(r, title=_title(f"Returns ({rc})", run_id))
    return _common_layout(fig, _title("Returns", run_id))


# -----------------------------
# 4) Rolling volatility
# -----------------------------
def rolling_vol_fig(
    ts: pd.DataFrame,
    run_id: Optional[str] = None,
    meta: Optional[dict] = None,
    window: int = 21,
    ann_factor: float = 252.0,
    ret_col: Optional[str] = None,
) -> go.Figure:
    ts = _ensure_dt_index(ts)
    rc = ret_col or _pick_first_col(ts, ("ret_net", "ret_gross", "ret"))
    if rc is None:
        fig = px.line(title=_title("Rolling Vol (missing returns)", run_id))
        return _common_layout(fig, _title("Rolling Vol", run_id))

    vol = _rolling_vol(ts[rc], window=window, ann_factor=ann_factor)
    fig = px.line(vol, title=_title(f"Rolling Vol ({window}d)", run_id))
    return _common_layout(fig, _title("Rolling Vol", run_id))


# -----------------------------
# 5) Rolling Sharpe
# -----------------------------
def rolling_sharpe_fig(
    ts: pd.DataFrame,
    run_id: Optional[str] = None,
    meta: Optional[dict] = None,
    window: int = 63,
    ann_factor: float = 252.0,
    ret_col: Optional[str] = None,
) -> go.Figure:
    ts = _ensure_dt_index(ts)
    rc = ret_col or _pick_first_col(ts, ("ret_net", "ret_gross", "ret"))
    if rc is None:
        fig = px.line(title=_title("Rolling Sharpe (missing returns)", run_id))
        return _common_layout(fig, _title("Rolling Sharpe", run_id))

    sh = _rolling_sharpe(ts[rc], window=window, ann_factor=ann_factor)
    fig = px.line(sh, title=_title(f"Rolling Sharpe ({window}d)", run_id))
    return _common_layout(fig, _title("Rolling Sharpe", run_id))


# -----------------------------
# 6) Weights / positions over time
#    - supports single-asset "weight" column
#    - or multi-asset weights DataFrame (columns = tickers)
# -----------------------------
def weights_fig(
    ts: pd.DataFrame,
    run_id: Optional[str] = None,
    meta: Optional[dict] = None,
    weight_cols: Optional[Sequence[str]] = None,
) -> go.Figure:
    ts = _ensure_dt_index(ts)

    if weight_cols:
        cols = [c for c in weight_cols if c in ts.columns]
    else:
        # common patterns:
        # - single asset: "weight"
        # - multi-asset: columns like "w_SPY", "w_XLE", etc.
        if "weight" in ts.columns:
            cols = ["weight"]
        else:
            cols = [c for c in ts.columns if c.startswith("w_")]

    if not cols:
        fig = px.line(title=_title("Weights (missing weight columns)", run_id))
        return _common_layout(fig, _title("Weights", run_id))

    if len(cols) == 1:
        fig = px.line(ts, y=cols[0], title=_title("Weight", run_id))
        return _common_layout(fig, _title("Weights", run_id))

    # multi-line
    df = ts[cols].astype(float)
    fig = px.line(df, title=_title("Weights", run_id))
    return _common_layout(fig, _title("Weights", run_id))


# -----------------------------
# 7) Regime probabilities (stacked area)
#    expects columns like p_state_0, p_state_1, ...
# -----------------------------
def regime_probs_fig(
    ts: pd.DataFrame,
    run_id: Optional[str] = None,
    meta: Optional[dict] = None,
    prob_prefix: str = "p_state_",
    labels: Optional[Sequence[str]] = None,
) -> go.Figure:
    ts = _ensure_dt_index(ts)

    pcols = [c for c in ts.columns if c.startswith(prob_prefix)]
    if not pcols:
        fig = px.line(title=_title("Regime Probabilities (missing p_state_* columns)", run_id))
        return _common_layout(fig, _title("Regime Probabilities", run_id))

    pcols = sorted(pcols)
    fig = go.Figure()
    for i, c in enumerate(pcols):
        name = labels[i] if (labels and i < len(labels)) else c
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts[c].astype(float),
            mode="lines",
            stackgroup="one",
            name=name,
        ))

    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return _common_layout(fig, _title("Regime Probabilities", run_id))


# -----------------------------
# 8) Forecast vs Realized (generic)
#    useful for your vol forecasting later:
#    - realized_col: "rv_xle"
#    - forecast_col: "rv_hat" or similar
# -----------------------------
def forecast_vs_realized_fig(
    ts: pd.DataFrame,
    run_id: Optional[str] = None,
    meta: Optional[dict] = None,
    realized_col: str = "realized",
    forecast_col: str = "forecast",
) -> go.Figure:
    ts = _ensure_dt_index(ts)

    if realized_col not in ts.columns or forecast_col not in ts.columns:
        fig = px.line(title=_title("Forecast vs Realized (missing columns)", run_id))
        return _common_layout(fig, _title("Forecast vs Realized", run_id))

    df = ts[[realized_col, forecast_col]].astype(float)
    fig = px.line(df, title=_title("Forecast vs Realized", run_id))
    return _common_layout(fig, _title("Forecast vs Realized", run_id))


# -----------------------------
# Optional: registry for scaling (if you want)
# -----------------------------
PLOT_REGISTRY: Dict[str, Any] = {
    "equity": equity_curve_fig,
    "drawdown": drawdown_fig,
    "returns": returns_fig,
    "rolling_vol": rolling_vol_fig,
    "rolling_sharpe": rolling_sharpe_fig,
    "weights": weights_fig,
    "regime_probs": regime_probs_fig,
    "forecast_vs_realized": forecast_vs_realized_fig,
}

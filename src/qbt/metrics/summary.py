from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def _to_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").dropna().astype(float)


def _equity_simple(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def _equity_log(r: pd.Series) -> pd.Series:
    return np.exp(r.cumsum())

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

def _perf_metrics(r: pd.Series, *, ann_factor: int, return_type: str, prefix: str) -> Dict[str, Any]:
    r = _to_series(r)
    if r.empty:
        return {f"{prefix}n_obs": 0}

    mean = float(r.mean())
    vol = float(r.std(ddof=1))
    sharpe = (mean / vol) * np.sqrt(ann_factor) if vol > 0 else np.nan

    if return_type == "simple":
        eq = _equity_simple(r)
    elif return_type == "log":
        eq = _equity_log(r)
    else:
        raise ValueError(f"return_type must be 'simple' or 'log', got {return_type}")

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min())
    cagr = float(eq.iloc[-1] ** (ann_factor / len(r)) - 1.0)

    return {
        f"{prefix}n_obs": int(len(r)),
        f"{prefix}mean_ann": mean * ann_factor,
        f"{prefix}vol_ann": vol * np.sqrt(ann_factor),
        f"{prefix}sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        f"{prefix}cagr": float(cagr) if np.isfinite(cagr) else None,
        f"{prefix}max_dd": max_dd,
        f"{prefix}ending_equity": float(eq.iloc[-1]),
    }


def _run_lengths_bool(x: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a boolean series x, return (values, lengths) for consecutive runs.
    """
    if x.empty:
        return np.array([], dtype=bool), np.array([], dtype=int)

    v = x.to_numpy(dtype=bool)
    # boundaries where value changes
    change = np.flatnonzero(np.diff(v, prepend=v[0]))
    # compute run starts
    starts = np.r_[0, change]
    # run ends are next start, last end is len(v)
    ends = np.r_[starts[1:], len(v)]
    lengths = ends - starts
    values = v[starts]
    return values, lengths


def _signal_metrics(sig: pd.Series, *, ann_factor: int, prefix: str = "signal_") -> Dict[str, Any]:
    """
    Metrics for a 0/1 signal:
    - turnover counts transitions (0->1 and 1->0)
    - avg holding period for state=1 and state=0 (run lengths)
    - pct_* fields as 0-100 (percent)
    """
    s = pd.to_numeric(sig, errors="coerce").fillna(0.0)
    s = (s > 0.5).astype(int)  # 0/1 ints

    n = int(len(s))
    if n == 0:
        return {f"{prefix}n_obs": 0}

    prev = s.shift(1).fillna(s.iloc[0]).astype(int)
    changed = (s != prev)

    n_turnovers = int(changed.sum())
    n_0_to_1 = int(((prev == 0) & (s == 1)).sum())
    n_1_to_0 = int(((prev == 1) & (s == 0)).sum())

    # Run lengths for each state (0 or 1)
    # Build group ids whenever state changes, then count run sizes
    run_id = changed.cumsum()
    run_sizes = s.groupby(run_id).size()               # length of each run
    run_state = s.groupby(run_id).first()              # 0/1 value for each run

    lens_1 = run_sizes[run_state == 1].to_numpy()
    lens_0 = run_sizes[run_state == 0].to_numpy()

    out: Dict[str, Any] = {
        f"{prefix}n_obs": n,
        f"{prefix}pct_state_1": float(s.mean() * 100.0),
        f"{prefix}pct_state_0": float((1.0 - s.mean()) * 100.0),
        f"{prefix}n_turnovers": n_turnovers,
        f"{prefix}n_0_to_1": n_0_to_1,
        f"{prefix}n_1_to_0": n_1_to_0,
        f"{prefix}turnovers_per_year": float(n_turnovers) * (ann_factor / max(n, 1)),
    }

    out[f"{prefix}avg_hold_state_1"] = float(lens_1.mean()) if lens_1.size else None
    out[f"{prefix}avg_hold_state_0"] = float(lens_0.mean()) if lens_0.size else None

    return out

def compute_portfolio_metrics(
    ts_df: pd.DataFrame,
    *,
    ann_factor: int = 252,
    return_type: str = "simple",  # set to "log" if your port_ret_* are log returns
    col_gross: str = "port_ret_gross",
    col_net: Optional[str] = "port_ret_net",
    col_bh: Optional[str] = "bh_ret",
    col_signal: Optional[str] = "signal",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if col_gross not in ts_df.columns:
        raise KeyError(f"ts_df missing required column: {col_gross}")

    # performance
    out.update(_perf_metrics(ts_df[col_gross], ann_factor=ann_factor, return_type=return_type, prefix="gross_"))

    if col_net and col_net in ts_df.columns:
        out.update(_perf_metrics(ts_df[col_net], ann_factor=ann_factor, return_type=return_type, prefix="net_"))

    if col_bh and col_bh in ts_df.columns:
        out.update(_perf_metrics(ts_df[col_bh], ann_factor=ann_factor, return_type=return_type, prefix="bh_"))

        # excess vs bh using net if available else gross
        ref = ts_df[col_net] if (col_net and col_net in ts_df.columns) else ts_df[col_gross]
        r_ref = _to_series(ref)
        r_bh = _to_series(ts_df[col_bh])
        idx = r_ref.index.intersection(r_bh.index)
        if len(idx) > 2:
            out.update(
                _perf_metrics(
                    r_ref.loc[idx] - r_bh.loc[idx],
                    ann_factor=ann_factor,
                    return_type=return_type,
                    prefix="excess_",
                )
            )
            net_sh = out.get("net_sharpe", out.get("gross_sharpe"))
            bh_sh = out.get("bh_sharpe")
            if (net_sh is not None) and (bh_sh is not None):
                out["sharpe_minus_bh"] = float(net_sh - bh_sh)

    # signal behavior
    if col_signal and col_signal in ts_df.columns:
        out.update(_signal_metrics(ts_df[col_signal], ann_factor=ann_factor, prefix="signal_"))

    # average / min / max for any weight column
    weight_cols = [c for c in ts_df.columns if str(c).endswith("weight")]

    for col in weight_cols:
        w = pd.to_numeric(ts_df[col], errors="coerce").fillna(0.0)
        base = str(col)

        out[f"avg_{base}"] = float(w.mean())
        out[f"max_{base}"] = float(w.max())
        out[f"min_{base}"] = float(w.min())
        # out[f"pct_invested_{base}"] = float((w != 0).mean() * 100.0)

    return out
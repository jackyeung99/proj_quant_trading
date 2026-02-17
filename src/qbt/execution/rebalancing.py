from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from qbt.core.types import Position

def latest_target_row(weights_ts: pd.DataFrame) -> pd.Series:
    """
    Extract the latest target weights as a Series indexed by symbol.
    Drops non-asset metadata columns if present.
    """
    if weights_ts is None or weights_ts.empty:
        raise ValueError("No target weights found in live store.")

    if not isinstance(weights_ts.index, pd.DatetimeIndex):
        raise ValueError("weights must be time-indexed (DatetimeIndex).")

    row = weights_ts.tail(1).iloc[0].copy()

    # Drop common metadata columns if you store them alongside weights
    meta_cols = {"generated_at_utc", "config_hash"}
    row = row.drop(labels=[c for c in row.index if c in meta_cols], errors="ignore")

    # keep only numeric
    row = pd.to_numeric(row, errors="coerce").fillna(0.0)

    # if a CASH column exists, keep it but you won't trade it
    return row


def normalize_weights(w: pd.Series, *, atol: float = 1e-6) -> pd.Series:
    """Ensure weights are non-negative and sum to 1 (unless all zero)."""
    w = w.astype(float).fillna(0.0)
    w[w < 0] = 0.0

    s = float(w.sum())
    if s <= atol:
        return w * 0.0
    return w / s


def positions_to_weights(
    positions: Dict[str, Position],
    *,
    equity_value: float,
) -> pd.Series:
    """
    Convert current positions to portfolio weights by market value.
    equity_value should be total account equity (or total portfolio value).
    """
    if equity_value <= 0:
        raise ValueError("equity_value must be > 0 to compute weights.")

    w = {}
    for sym, p in positions.items():
        # long-only MVP
        mv = float(p.market_value)
        w[sym] = mv / equity_value
    return pd.Series(w, dtype=float).fillna(0.0)


def compute_target_dollars(
    target_w: pd.Series,
    *,
    equity_value: float,
) -> pd.Series:
    """
    Convert target weights into target dollar exposures.
    """
    return target_w * float(equity_value)


def compute_trade_dollars(
    target_dollars: pd.Series,
    current_dollars: pd.Series,
    *,
    min_trade_dollars: float = 5.0,
) -> pd.Series:
    """
    Desired $ change per symbol = target - current.
    Filters small trades.
    """
    all_syms = target_dollars.index.union(current_dollars.index)
    tgt = target_dollars.reindex(all_syms).fillna(0.0)
    cur = current_dollars.reindex(all_syms).fillna(0.0)
    delta = (tgt - cur)

    # filter noise
    delta = delta[delta.abs() >= float(min_trade_dollars)]
    return delta.sort_values()



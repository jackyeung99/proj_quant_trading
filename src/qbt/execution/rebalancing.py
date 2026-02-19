from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import uuid

import pandas as pd

from qbt.core.logging import get_logger

from qbt.storage.artifacts  import LiveStore
from qbt.execution.alpaca_client import AlpacaTradingAPI
from qbt.execution.orders import build_qty_orders
from qbt.execution.weight_math import (
    compute_target_dollars, compute_target_shares, compute_trade_shares, latest_target_row
)

logger = get_logger(__name__)


META_COLS = {
    "asof_utc",
    "generated_at_utc",
    "config_hash",
    "market_tz",
    "cutoff_hour",
}

CASH_LIKE = {"CASH", "USD"}  # keep if you handle cash separately

def _snapshot_portfolio(client: AlpacaTradingAPI) -> tuple[float, pd.Series, pd.Series]:
    equity = float(client.get_equity())
    pos = client.get_active_positions()

    current_shares = pd.Series({sym: float(p.qty) for sym, p in pos.items()}, dtype=float).fillna(0.0)
    current_dollars = pd.Series({sym: float(p.market_value) for sym, p in pos.items()}, dtype=float).fillna(0.0)
    return equity, current_shares, current_dollars



def _extract_weight_assets(latest_row: pd.Series) -> list[str]:
    """
    Identify tradable assets from a latest weight row.

    Rule:
      - Only numeric entries are considered weights
      - Explicitly exclude known meta cols and cash-like columns
    """
    assets = []
    for k, v in latest_row.items():
        if k in META_COLS or k in CASH_LIKE:
            continue
        # numeric weights only
        if pd.api.types.is_number(v):
            assets.append(k)
    return assets


def _load_target_weights(storage: LiveStore, *, strat: str, universe: str):
    weights_ts = storage.read_weights(strategy=strat, universe=universe)
    if weights_ts is None or weights_ts.empty:
        return None, None, None

    # Prefer a deterministic timestamp if present; otherwise fall back to index max
    if "asof_utc" in weights_ts.columns:
        try:
            asof_val = pd.to_datetime(weights_ts["asof_utc"]).max()
            asof = str(asof_val)
        except Exception:
            asof = str(weights_ts.index.max())
    else:
        asof = str(weights_ts.index.max())

    target_row = latest_target_row(weights_ts)  # expects a Series

    # Build target weights using only valid asset columns
    target_assets = _extract_weight_assets(target_row)
    target_w = target_row.reindex(target_assets).astype(float).fillna(0.0)

    return weights_ts, asof, target_w

def _load_prices(
    client: AlpacaTradingAPI,
    symbols: list[str],
) -> pd.Series:
    """
    Fetch latest prices for symbols.

    Returns
    -------
    pd.Series
        Index: symbol
        Values: float price (> 0)

    Notes
    -----
    • Drops symbols with missing/invalid prices
    • Removes inf/NaN
    • Safe for fractional trading logic
    """

    if not symbols:
        return pd.Series(dtype=float)

    # Deduplicate while preserving order
    unique_symbols = list(dict.fromkeys(symbols))

    # ---- Broker call ----
    px = client.get_latest_prices(unique_symbols)

    if not px:
        return pd.Series(dtype=float)

    # ---- Convert to Series ----
    s = pd.Series(px, dtype=float)

    # ---- Clean invalid values ----
    s = s.replace([float("inf"), -float("inf")], pd.NA)
    s = s.dropna()

    # Remove non-positive prices
    s = s[s > 0]

    # ---- Ensure index order matches input ----
    s = s.reindex(unique_symbols).dropna()

    return s


def _maybe_liquidate_for_holding_period(
    client: AlpacaTradingAPI,
    *,
    dry_run: bool,
    enabled: bool,
    mode: str,
    tif: str,
    symbols,
    include_shorts: bool,
    skip_if_open_order: bool,
) -> dict | None:
    """
    Returns hp_result dict if liquidation ran (or was skipped due to dry_run),
    otherwise None.
    """
    if not enabled:
        return None

    if mode not in ("liquidate", "liquidate_then_rebalance", "rebalance_then_liquidate"):
        # ignore unknown modes rather than exploding in prod; you can tighten later
        logger.warning(f"Unknown holding_period.mode={mode!r} — skipping holding-period behavior")
        return None

    if dry_run:
        logger.warning("Dry run — holding_period liquidation skipped")
        return {
            "submitted": [],
            "skipped": [{"symbol": "*", "reason": "dry_run"}],
            "errors": [],
            "tif": tif,
            "mode": mode,
        }

    at_open = tif == "opg"
    at_close = tif == "cls"

    results = {"submitted": [], "skipped": [], "errors": []}

    if symbols is None:
        out = client.liquidate(
            at_open=at_open,
            at_close=at_close,
            include_shorts=include_shorts,
            skip_if_open_order=skip_if_open_order,
            client_order_id_prefix=f"hp_{tif}",
        )
        results["submitted"] += out.get("submitted", [])
        results["skipped"] += out.get("skipped", [])
        results["errors"] += out.get("errors", [])
    else:
        for sym in symbols:
            out = client.liquidate(
                symbol=str(sym),
                at_open=at_open,
                at_close=at_close,
                include_shorts=include_shorts,
                skip_if_open_order=skip_if_open_order,
                client_order_id_prefix=f"hp_{tif}",
            )
            results["submitted"] += out.get("submitted", [])
            results["skipped"] += out.get("skipped", [])
            results["errors"] += out.get("errors", [])

    logger.info(
        f"Holding-period liquidation | tif={tif} "
        f"submitted={len(results['submitted'])} skipped={len(results['skipped'])} errors={len(results['errors'])}"
    )
    return {**results, "tif": tif, "mode": mode}


def _write_planned_orders(
    storage: LiveStore,
    *,
    strat: str,
    universe: str,
    orders: list[dict],
    trade_dollars: pd.Series,
    equity: float,
    asof: str,
    run_id: str,
    exec_id: str,
    dry_run: bool,
    hp_enabled: bool,
    hp_mode: str,
    hp_tif: str,
) -> tuple[str, str, pd.DataFrame]:
    batch_id = uuid.uuid4().hex[:12]
    now_utc = pd.Timestamp.utcnow()

    orders_df = pd.DataFrame(
        [
            {
                "timestamp": now_utc,
                "batch_id": batch_id,
                "run_id": run_id,
                "exec_id": exec_id,
                "strategy": strat,
                "universe": universe,
                "asof": asof,
                "symbol": o["symbol"],
                "side": o["side"],
                "notional": float(o["notional"]),
                "trade_dollars": float(trade_dollars.get(o["symbol"], float("nan"))),
                "equity": float(equity),
                "dry_run": bool(dry_run),
                "holding_period_enabled": bool(hp_enabled),
                "holding_period_mode": hp_mode if hp_enabled else "",
                "holding_period_tif": hp_tif if hp_enabled else "",
            }
            for o in orders
        ]
    )

    orders_key = storage.write_orders_batch(
        strategy=strat,
        universe=universe,
        orders=orders_df,
        batch_id=batch_id,
    )
    return batch_id, orders_key, orders_df


@dataclass(frozen=True)
class RebalancePlan:
    prices: pd.Series
    target_dollars: pd.Series
    target_shares: pd.Series
    trade_shares: pd.Series
    orders: list[dict]
    gross_notional: float


def plan_rebalance(
    *,
    target_w: pd.Series,
    equity: float,
    current_shares: pd.Series,
    prices: pd.Series,
    min_trade_dollars: float,
) -> RebalancePlan:

    target_dollars = compute_target_dollars(
        target_w,
        equity_value=equity,
    )

    target_dollars = target_dollars.reindex(prices.index).fillna(0.0)

 
    target_shares = compute_target_shares(
        target_dollars,
        prices,
    )

    trade_shares = compute_trade_shares(
        target_shares=target_shares,
        current_shares=current_shares.reindex(target_shares.index).fillna(0.0),
        prices=prices,
        min_trade_dollars=min_trade_dollars,
    )

    orders = build_qty_orders(
        trade_shares,
        current_shares=current_shares,
    )

    gross_notional = float(
        (trade_shares.abs() * prices.reindex(trade_shares.index)).sum()
    )

    return RebalancePlan(
        prices=prices,
        target_dollars=target_dollars,
        target_shares=target_shares,
        trade_shares=trade_shares,
        orders=orders,
        gross_notional=gross_notional,
    )
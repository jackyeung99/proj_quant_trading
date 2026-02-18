from __future__ import annotations

from typing import Optional
import pandas as pd
import math
import uuid 

from qbt.core.logging import get_logger
from qbt.execution.alpaca_client import AlpacaTradingAPI


logger = get_logger(__name__)

def floor_to_step(qty: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    return math.floor(qty / step) * step

def safe_round_qty(qty: float, *, step: float = 1e-6) -> float:
    """Always rounds toward 0 in a safe way."""
    if qty == 0:
        return 0.0
    s = 1.0 if qty > 0 else -1.0
    return float(s * floor_to_step(abs(qty), step))



def build_qty_orders(
    trade_shares: pd.Series,
    *,
    current_shares: pd.Series | None = None,
    min_qty: float = 1e-6,   # ignore dust
    qty_step: float = 1e-6,
) -> list[dict]:

    orders: list[dict] = []

    for sym, dq in trade_shares.items():
        
        qty = safe_round_qty(dq, step=qty_step)

        if abs(qty) < min_qty:
            continue

        side = "buy" if qty > 0 else "sell"

        orders.append(
            {
                "symbol": sym,
                "side": side,
                "qty": abs(qty), 
            }
        )

    return orders

def _build_orders_from_trade_dollars(trade_dollars: pd.Series) -> list[dict]:
    orders = []
    for sym, d in trade_dollars.items():
        side = "buy" if d > 0 else "sell"
        notional = round(float(abs(d)), 2)
        orders.append({"symbol": sym, "side": side, "notional": notional})
    return orders


def _submit_orders(
    client: AlpacaTradingAPI,
    orders: list[dict],
) -> None:

    # -------------------------
    # Submit SELLS first
    # -------------------------
    for o in (x for x in orders if x["side"] == "sell"):

        logger.info(
            f"Submitting SELL | symbol={o['symbol']} qty={o['qty']:.6f}"
        )

        client.place_order(
            symbol=o["symbol"],
            side="sell",
            qty=o["qty"],
        )

    # -------------------------
    # Submit BUYS second
    # -------------------------
    for o in (x for x in orders if x["side"] == "buy"):

        logger.info(
            f"Submitting BUY  | symbol={o['symbol']} qty={o['qty']:.6f}"
        )

        client.place_order(
            symbol=o["symbol"],
            side="buy",
            qty=o["qty"],
        )
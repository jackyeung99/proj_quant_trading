from __future__ import annotations

from typing import Optional
import pandas as pd
import uuid 

from qbt.core.logging import get_logger
from qbt.execution.alpaca_client import AlpacaTradingAPI


logger = get_logger(__name__)

def build_qty_orders(
    trade_shares: pd.Series,
    *,
    current_shares: pd.Series | None = None,
    min_qty: float = 1e-6,   # ignore dust
) -> list[dict]:

    orders: list[dict] = []

    for sym, dq in trade_shares.items():

        qty = float(dq)

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
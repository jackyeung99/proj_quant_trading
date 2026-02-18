# test/execution/test_orders.py
from __future__ import annotations

import pandas as pd
import pytest

from qbt.execution.orders import build_qty_orders, _submit_orders, safe_round_qty, floor_to_step


# ---------------------------------------------------------------------
# build_qty_orders tests
# ---------------------------------------------------------------------

def test_build_qty_orders_basic_buy_sell():
    trade_shares = pd.Series({"XLE": 10.0, "SPY": -3.0})

    orders = build_qty_orders(trade_shares)

    assert {"symbol": "XLE", "side": "buy", "qty": pytest.approx(10.0)} in orders
    assert {"symbol": "SPY", "side": "sell", "qty": pytest.approx(3.0)} in orders
    assert len(orders) == 2


def test_build_qty_orders_skips_dust_below_min_qty():
    trade_shares = pd.Series({"XLE": 5e-7, "SPY": -4e-7, "QQQ": 2e-6})

    orders = build_qty_orders(trade_shares, min_qty=1e-6, qty_step=1e-6)

    # XLE and SPY should be skipped (abs(qty) < min_qty)
    assert all(o["symbol"] != "XLE" for o in orders)
    assert all(o["symbol"] != "SPY" for o in orders)

    # QQQ survives
    assert {"symbol": "QQQ", "side": "buy", "qty": pytest.approx(2e-6)} in orders
    assert len(orders) == 1


def test_build_qty_orders_rounds_toward_zero_to_step():
    # qty_step=0.01 means we floor magnitude to 2 decimals (toward 0)
    trade_shares = pd.Series({"XLE": 1.239, "SPY": -2.999})

    orders = build_qty_orders(trade_shares, qty_step=0.01, min_qty=0.0)

    # 1.239 -> 1.23, -2.999 -> -2.99 (rounded toward 0)
    assert {"symbol": "XLE", "side": "buy", "qty": pytest.approx(1.23)} in orders
    assert {"symbol": "SPY", "side": "sell", "qty": pytest.approx(2.99)} in orders


def test_build_qty_orders_handles_zero_and_missing_symbols_cleanly():
    trade_shares = pd.Series({"XLE": 0.0, "SPY": 1.0})

    orders = build_qty_orders(trade_shares)

    assert len(orders) == 1
    assert orders[0]["symbol"] == "SPY"
    assert orders[0]["side"] == "buy"
    assert orders[0]["qty"] == pytest.approx(1.0)


def test_build_qty_orders_deterministic_order_matches_series_iteration():
    # pandas preserves insertion order for dict -> Series
    trade_shares = pd.Series({"B": 1.0, "A": -1.0, "C": 2.0})
    orders = build_qty_orders(trade_shares, min_qty=0.0)

    assert [o["symbol"] for o in orders] == ["B", "A", "C"]
    assert [o["side"] for o in orders] == ["buy", "sell", "buy"]


# ---------------------------------------------------------------------
# Helper rounding tests (optional but useful)
# ---------------------------------------------------------------------

def test_floor_to_step():
    assert floor_to_step(1.239, 0.01) == pytest.approx(1.23)
    assert floor_to_step(2.0, 0.5) == pytest.approx(2.0)
    assert floor_to_step(2.49, 0.5) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        floor_to_step(1.0, 0.0)


def test_safe_round_qty_rounds_toward_zero():
    assert safe_round_qty(1.239, step=0.01) == pytest.approx(1.23)
    assert safe_round_qty(-1.239, step=0.01) == pytest.approx(-1.23)
    assert safe_round_qty(0.0, step=0.01) == 0.0


# ---------------------------------------------------------------------
# _submit_orders tests (SELLS first, then BUYS)
# ---------------------------------------------------------------------

class DummyClient:
    def __init__(self):
        self.calls = []

    def place_order(self, *, symbol: str, side: str, qty: float):
        self.calls.append({"symbol": symbol, "side": side, "qty": qty})


def test_submit_orders_sells_before_buys_even_if_mixed_input_order():
    client = DummyClient()

    orders = [
        {"symbol": "XLE", "side": "buy", "qty": 1.0},
        {"symbol": "SPY", "side": "sell", "qty": 2.0},
        {"symbol": "QQQ", "side": "buy", "qty": 3.0},
        {"symbol": "IWM", "side": "sell", "qty": 4.0},
    ]

    _submit_orders(client, orders)

    # Expect sells in their relative order first, then buys in their relative order
    assert client.calls == [
        {"symbol": "SPY", "side": "sell", "qty": 2.0},
        {"symbol": "IWM", "side": "sell", "qty": 4.0},
        {"symbol": "XLE", "side": "buy", "qty": 1.0},
        {"symbol": "QQQ", "side": "buy", "qty": 3.0},
    ]


def test_submit_orders_only_sells():
    client = DummyClient()
    orders = [
        {"symbol": "SPY", "side": "sell", "qty": 2.5},
        {"symbol": "XLE", "side": "sell", "qty": 1.0},
    ]

    _submit_orders(client, orders)

    assert client.calls == [
        {"symbol": "SPY", "side": "sell", "qty": 2.5},
        {"symbol": "XLE", "side": "sell", "qty": 1.0},
    ]


def test_submit_orders_only_buys():
    client = DummyClient()
    orders = [
        {"symbol": "SPY", "side": "buy", "qty": 2.5},
        {"symbol": "XLE", "side": "buy", "qty": 1.0},
    ]

    _submit_orders(client, orders)

    assert client.calls == [
        {"symbol": "SPY", "side": "buy", "qty": 2.5},
        {"symbol": "XLE", "side": "buy", "qty": 1.0},
    ]

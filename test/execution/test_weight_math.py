import numpy as np
import pandas as pd
import pytest

from qbt.execution.weight_math import (
    compute_target_dollars,
    compute_target_shares,
    compute_trade_shares,
)

from qbt.execution.orders import build_qty_orders
from qbt.execution.rebalancing import plan_rebalance


def _empty_shares() -> pd.Series:
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------
# End-to-end tests (plan_rebalance)
# ---------------------------------------------------------------------

def test_plan_rebalance_single_asset_from_cash_only():
    equity = 10_000.0
    target_w = pd.Series({"XLE": 0.50})
    prices = pd.Series({"XLE": 50.0})
    current_shares = _empty_shares()

    plan = plan_rebalance(
        target_w=target_w,
        equity=equity,
        current_shares=current_shares,
        prices=prices,
        min_trade_dollars=2.0,
    )

    assert plan is not None
    assert isinstance(plan.orders, list)
    assert len(plan.orders) == 1

    # Targets
    assert plan.target_dollars.loc["XLE"] == pytest.approx(5_000.0)
    assert plan.target_shares.loc["XLE"] == pytest.approx(100.0)

    # Trades and orders
    assert plan.trade_shares.loc["XLE"] == pytest.approx(100.0)
    assert plan.orders[0] == {"symbol": "XLE", "side": "buy", "qty": 100.0}

    # Gross notional = |100| * 50
    assert plan.gross_notional == pytest.approx(5_000.0)


def test_plan_rebalance_multi_asset_buy_and_sell():
    """
    Start with some positions, rebalance into a new target.
    Includes BOTH buy and sell.
    """
    equity = 10_000.0

    # target: 60% XLE, 40% SPY
    target_w = pd.Series({"XLE": 0.60, "SPY": 0.40})

    prices = pd.Series({"XLE": 50.0, "SPY": 100.0})

    # currently: 50 XLE (=$2500), 70 SPY (=$7000) -> total=$9500 notional
    current_shares = pd.Series({"XLE": 50.0, "SPY": 70.0})

    plan = plan_rebalance(
        target_w=target_w,
        equity=equity,
        current_shares=current_shares,
        prices=prices,
        min_trade_dollars=1.0,
    )

    # Target dollars
    assert plan.target_dollars.loc["XLE"] == pytest.approx(6_000.0)
    assert plan.target_dollars.loc["SPY"] == pytest.approx(4_000.0)

    # Target shares
    assert plan.target_shares.loc["XLE"] == pytest.approx(120.0)  # 6000/50
    assert plan.target_shares.loc["SPY"] == pytest.approx(40.0)   # 4000/100

    # Trade shares = target - current
    assert plan.trade_shares.loc["XLE"] == pytest.approx(70.0)    # buy 70
    assert plan.trade_shares.loc["SPY"] == pytest.approx(-30.0)   # sell 30

    # Orders should contain both sides (order order may vary; assert by content)
    assert {"symbol": "XLE", "side": "buy", "qty": 70.0} in plan.orders
    assert {"symbol": "SPY", "side": "sell", "qty": 30.0} in plan.orders

    # Gross notional = 70*50 + 30*100 = 3500 + 3000 = 6500
    assert plan.gross_notional == pytest.approx(6_500.0)


def test_plan_rebalance_min_trade_dollars_filters_small_trades():
    """
    If the dollar value of a trade is below min_trade_dollars, it should be skipped.
    """
    equity = 100.0
    target_w = pd.Series({"XLE": 1.0, "SPY": 0.0})
    prices = pd.Series({"XLE": 50.0, "SPY": 100.0})

    # already basically there: 2 XLE shares = $100
    # target is also $100 in XLE => trade is 0
    current_shares = pd.Series({"XLE": 2.0, "SPY": 0.0})

    plan = plan_rebalance(
        target_w=target_w,
        equity=equity,
        current_shares=current_shares,
        prices=prices,
        min_trade_dollars=10.0,
    )

    assert "XLE" not in plan.trade_shares.index
    assert len(plan.orders) == 0
    assert plan.gross_notional == pytest.approx(0.0)


def test_plan_rebalance_aligns_to_prices_and_fills_missing_weights_with_zero():
    """
    target_w missing a symbol that exists in prices => should be treated as 0 weight.
    """
    equity = 1_000.0
    target_w = pd.Series({"XLE": 1.0})  # SPY missing => 0
    prices = pd.Series({"XLE": 50.0, "SPY": 100.0})
    current_shares = pd.Series({"XLE": 0.0, "SPY": 10.0})  # currently long SPY

    plan = plan_rebalance(
        target_w=target_w,
        equity=equity,
        current_shares=current_shares,
        prices=prices,
        min_trade_dollars=1.0,
    )

    # SPY target dollars should be 0 due to missing weight
    assert plan.target_dollars.loc["SPY"] == pytest.approx(0.0)
    assert plan.target_shares.loc["SPY"] == pytest.approx(0.0)

    # Since we currently hold 10 SPY, should plan to sell them
    assert plan.trade_shares.loc["SPY"] == pytest.approx(-10.0)
    assert {"symbol": "SPY", "side": "sell", "qty": 10.0} in plan.orders


# ---------------------------------------------------------------------
# Step-level unit tests (each component inside plan_rebalance)
# ---------------------------------------------------------------------

def test_compute_target_dollars_sums_to_equity_when_weights_sum_to_one():
    equity = 10_000.0
    w = pd.Series({"A": 0.2, "B": 0.3, "C": 0.5})

    td = compute_target_dollars(w, equity_value=equity)

    assert float(td.sum()) == pytest.approx(equity)
    assert td.loc["A"] == pytest.approx(2_000.0)
    assert td.loc["B"] == pytest.approx(3_000.0)
    assert td.loc["C"] == pytest.approx(5_000.0)


def test_compute_target_shares_divides_by_prices_and_preserves_index():
    td = pd.Series({"A": 2_000.0, "B": 3_000.0})
    prices = pd.Series({"A": 50.0, "B": 100.0})

    ts = compute_target_shares(td, prices)

    assert list(ts.index) == ["A", "B"]
    assert ts.loc["A"] == pytest.approx(40.0)
    assert ts.loc["B"] == pytest.approx(30.0)


def test_compute_trade_shares_basic_and_min_trade_filtering():
    target_shares = pd.Series({"A": 10.0, "B": 1.0})
    current_shares = pd.Series({"A": 7.0, "B": 1.5})
    prices = pd.Series({"A": 100.0, "B": 2.0})

    # A trade is 3 shares => $300 (kept)
    # B trade is -0.5 shares => $1 (filtered if min_trade_dollars > 1)
    trade = compute_trade_shares(
        target_shares=target_shares,
        current_shares=current_shares,
        prices=prices,
        min_trade_dollars=2.0,
    )


    assert trade.loc["A"] == pytest.approx(3.0)
    assert "B" not in trade.index


def test_build_qty_orders_creates_buy_sell_and_skips_zeros():
    trade_shares = pd.Series({"A": 3.0, "B": -2.0, "C": 0.0})
    current_shares = pd.Series({"A": 0.0, "B": 10.0, "C": 5.0})

    orders = build_qty_orders(trade_shares, current_shares=current_shares)

    assert {"symbol": "A", "side": "buy", "qty": 3.0} in orders
    assert {"symbol": "B", "side": "sell", "qty": 2.0} in orders
    assert all(o["symbol"] != "C" for o in orders)


def test_plan_rebalance_gross_notional_matches_trade_shares_times_prices():
    equity = 5_000.0
    target_w = pd.Series({"A": 0.5, "B": 0.5})
    prices = pd.Series({"A": 10.0, "B": 20.0})
    current_shares = pd.Series({"A": 0.0, "B": 0.0})

    plan = plan_rebalance(
        target_w=target_w,
        equity=equity,
        current_shares=current_shares,
        prices=prices,
        min_trade_dollars=1.0,
    )

    expected_gross = float((plan.trade_shares.abs() * prices.reindex(plan.trade_shares.index)).sum())
    assert plan.gross_notional == pytest.approx(expected_gross)

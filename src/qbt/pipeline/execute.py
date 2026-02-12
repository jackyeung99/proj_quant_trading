from qbt.execution.alpaca_client import AlpacaTradingAPI
from qbt.execution.rebalancing import *

from qbt.core.logging import get_logger
from qbt.storage.artifacts import LiveStore

import pandas as pd


logger = get_logger(__name__)


def execute_weights(storage: LiveStore, execution_cfg: dict) -> dict:
    """
    Read latest target weights from live store and rebalance using Alpaca.

    Returns a dict summary (useful for logs/dashboards).
    """
    # ------------------------------------------------------------------
    # Config + guards
    # ------------------------------------------------------------------
    strat_name = execution_cfg.get("strategy_name")
    universe = execution_cfg.get("universe", "")
    min_trade_dollars = float(execution_cfg.get("min_trade_dollars", 25.0))
    dry_run = bool(execution_cfg.get("dry_run", True))

    if not strat_name:
        raise ValueError("execution_cfg must include 'strategy_name'.")

    logger.info(
        "Starting execution step",
        extra={
            "strategy": strat_name,
            "universe": universe,
            "dry_run": dry_run,
            "min_trade_dollars": min_trade_dollars,
        },
    )

    # ------------------------------------------------------------------
    # Alpaca client
    # ------------------------------------------------------------------
    client = AlpacaTradingAPI(cfg=execution_cfg.get("alpaca", {}) or {})
    logger.debug("Alpaca client initialized")

    # ------------------------------------------------------------------
    # 1) Load latest target weights
    # ------------------------------------------------------------------
    weights_ts = storage.read_weights(strategy=strat_name, universe=universe)

    if weights_ts.empty:
        logger.warning(
            "No target weights found; skipping execution",
            extra={"strategy": strat_name, "universe": universe},
        )
        return {
            "strategy_name": strat_name,
            "universe": universe,
            "orders": [],
            "dry_run": dry_run,
            "reason": "no_weights",
        }

    target_row = latest_target_row(weights_ts)

    cash_like = {"CASH", "USD"}
    target_assets = [c for c in target_row.index if c not in cash_like]
    target_w = target_row.reindex(target_assets).fillna(0.0)

    logger.info(
        "Loaded target weights",
        extra={
            "strategy": strat_name,
            "universe": universe,
            "asof": str(weights_ts.index.max()),
            "target_weights": target_w.to_dict(),
        },
    )

    # ------------------------------------------------------------------
    # 2) Current portfolio state
    # ------------------------------------------------------------------
    equity = float(client.get_equity())
    pos = client.get_active_positions()

    current_mv = {sym: float(p.market_value) for sym, p in pos.items()}
    current_dollars = pd.Series(current_mv, dtype=float).fillna(0.0)

    logger.info(
        "Fetched current portfolio state",
        extra={
            "equity": equity,
            "positions": current_dollars.to_dict(),
        },
    )

    # ------------------------------------------------------------------
    # 3) Compute target and trade deltas
    # ------------------------------------------------------------------
    target_dollars = compute_target_dollars(target_w, equity_value=equity)

    trade_dollars = compute_trade_dollars(
        target_dollars=target_dollars,
        current_dollars=current_dollars,
        min_trade_dollars=min_trade_dollars,
    )

    if trade_dollars.empty:
        logger.info(
            "No trades required (below threshold or already aligned)",
            extra={"strategy": strat_name, "universe": universe},
        )
        return {
            "strategy_name": strat_name,
            "universe": universe,
            "equity": equity,
            "target_weights": target_w.to_dict(),
            "orders": [],
            "dry_run": dry_run,
        }

    logger.info(
        "Computed trade deltas",
        extra={
            "trade_dollars": trade_dollars.to_dict(),
        },
    )

    # ------------------------------------------------------------------
    # 4) Build orders
    # ------------------------------------------------------------------
    orders = []
    for sym, d in trade_dollars.items():
        side = "buy" if d > 0 else "sell"
        notional = round(float(abs(d)), 2)
        orders.append({"symbol": sym, "side": side, "notional": notional})

    logger.info(
        "Prepared orders",
        extra={
            "num_orders": len(orders),
            "orders": orders,
        },
    )


    # ------------------------------------------------------------------
    # 5) Dry run or live execution
    # ------------------------------------------------------------------
    if dry_run:
        logger.warning(
            "Dry run enabled — orders NOT sent",
            extra={"orders": orders},
        )
        return {
            "strategy_name": strat_name,
            "universe": universe,
            "asof": str(weights_ts.index.max()),
            "equity": equity,
            "target_weights": target_w.to_dict(),
            "orders": orders,
            "dry_run": True,
        }

    # Execute sells first
    for o in [x for x in orders if x["side"] == "sell"]:
        logger.info("Submitting SELL order", extra=o)
        client.place_order(
            symbol=o["symbol"],
            side="sell",
            notional=o["notional"],
        )

    # Execute buys after
    for o in [x for x in orders if x["side"] == "buy"]:
        logger.info("Submitting BUY order", extra=o)
        client.place_order(
            symbol=o["symbol"],
            side="buy",
            notional=o["notional"],
        )

    logger.info(
        "Execution complete",
        extra={
            "strategy": strat_name,
            "universe": universe,
            "num_orders": len(orders),
        },
    )

    return {
        "strategy_name": strat_name,
        "universe": universe,
        "asof": str(weights_ts.index.max()),
        "equity": equity,
        "target_weights": target_w.to_dict(),
        "orders": orders,
        "dry_run": False,
    }

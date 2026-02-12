from qbt.execution.alpaca_client import AlpacaTradingAPI
from qbt.execution.rebalancing import *

from qbt.core.logging import get_logger
from qbt.storage.artifacts import LiveStore
import uuid

import pandas as pd

logger = get_logger(__name__)


def execute_weights(storage: LiveStore, execution_cfg: dict) -> dict:
    """
    Read latest target weights and rebalance using Alpaca.

    Idempotency/guards (MVP):
      1) best-effort lock (prevents concurrent runs)
      2) skip if already executed this asof
      3) (optional) skip if open orders exist for symbols (requires Alpaca client support)
      4) write planned orders + trades ledger + last_exec
    """
    # ------------------------------------------------------------------
    # Config + guards
    # ------------------------------------------------------------------
    strat_name = execution_cfg.get("strategy_name")
    universe = execution_cfg.get("universe", "")
    min_trade_dollars = float(execution_cfg.get("min_trade_dollars", 25.0))
    dry_run = bool(execution_cfg.get("dry_run", True))

    # guard knobs
    lock_enabled = bool(execution_cfg.get("lock_enabled", True))
    allow_replay_same_asof = bool(execution_cfg.get("allow_replay_same_asof", False))
    check_open_orders = bool(execution_cfg.get("check_open_orders", True))

    if not strat_name:
        raise ValueError("execution_cfg must include 'strategy_name'.")

    run_id = str(execution_cfg.get("run_id") or uuid.uuid4().hex[:12])
    exec_id = uuid.uuid4().hex[:12]  # unique attempt id for logging/audit

    logger.info(
        f"Execution start | strategy={strat_name} universe={universe} "
        f"dry_run={dry_run} min_trade_dollars={min_trade_dollars:.2f} run_id={run_id} exec_id={exec_id}"
    )

    # ------------------------------------------------------------------
    # Best-effort lock (prevents double-submit from concurrent runs)
    # ------------------------------------------------------------------
    if lock_enabled and not dry_run:
        existing = storage.read_lock(strategy=strat_name, universe=universe)
        
        if existing.get("locked", False):
            logger.warning(f"Execution locked | strategy={strat_name} universe={universe} lock={existing}")
            return {
                "strategy_name": strat_name,
                "universe": universe,
                "orders": [],
                "dry_run": dry_run,
                "reason": "locked",
                "lock": existing,
            }

        storage.write_lock(
            strategy=strat_name,
            universe=universe,
            meta={
                "locked": True,
                "run_id": run_id,
                "exec_id": exec_id,
                "created_at": pd.Timestamp.utcnow().isoformat(),
            },
        )

    try:
        # ------------------------------------------------------------------
        # Alpaca client
        # ------------------------------------------------------------------
        client = AlpacaTradingAPI(cfg=execution_cfg.get("alpaca", {}) or {})
        logger.debug("Alpaca client initialized")

        # ------------------------------------------------------------------
        # 1) Load latest target weights (single-row df)
        # ------------------------------------------------------------------
        weights_ts = storage.read_weights(strategy=strat_name, universe=universe)
        if weights_ts.empty:
            logger.warning(f"No target weights found | strategy={strat_name} universe={universe}")
            return {
                "strategy_name": strat_name,
                "universe": universe,
                "orders": [],
                "dry_run": dry_run,
                "reason": "no_weights",
            }

        # asof identity for idempotency
        asof = str(weights_ts.index.max())

        # skip if already executed this asof (unless explicitly allowed)
        if (not allow_replay_same_asof) and storage.already_executed_asof(strategy=strat_name, universe=universe, asof=asof):
            last_exec = storage.read_last_exec(strategy=strat_name, universe=universe)
            logger.warning(f"Skip (already executed asof) | asof={asof} last_exec={last_exec}")
            return {
                "strategy_name": strat_name,
                "universe": universe,
                "asof": asof,
                "orders": [],
                "dry_run": dry_run,
                "reason": "already_executed_asof",
                "last_exec": last_exec,
            }

        target_row = latest_target_row(weights_ts)

        cash_like = {"CASH", "USD"}
        target_assets = [c for c in target_row.index if c not in cash_like]
        target_w = target_row.reindex(target_assets).fillna(0.0)

        nonzero_w = int((target_w.abs() > 1e-12).sum())
        logger.info(f"Loaded target weights | asof={asof} assets={len(target_w)} nonzero={nonzero_w}")

        # ------------------------------------------------------------------
        # 2) Current portfolio state
        # ------------------------------------------------------------------
        equity = float(client.get_equity())
        pos = client.get_active_positions()

        current_mv = {sym: float(p.market_value) for sym, p in pos.items()}
        current_dollars = pd.Series(current_mv, dtype=float).fillna(0.0)

        logger.info(f"Portfolio state | equity={equity:.2f} positions={len(current_dollars)}")

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
            logger.info(f"No trades required | strategy={strat_name} universe={universe}")
            return {
                "strategy_name": strat_name,
                "universe": universe,
                "asof": asof,
                "equity": equity,
                "target_weights": target_w.to_dict(),
                "orders": [],
                "dry_run": dry_run,
                "reason": "no_trades",
            }

        n_trades = int(trade_dollars.shape[0])
        gross = float(trade_dollars.abs().sum())
        logger.info(f"Trade deltas computed | n={n_trades} gross_notional={gross:.2f}")

        # ------------------------------------------------------------------
        # 4) Build orders
        # ------------------------------------------------------------------
        orders = []
        for sym, d in trade_dollars.items():
            side = "buy" if d > 0 else "sell"
            notional = round(float(abs(d)), 2)
            orders.append({"symbol": sym, "side": side, "notional": notional})

        logger.info(f"Prepared orders | n={len(orders)}")

        # ------------------------------------------------------------------
        # 4b) Write planned orders batch (even in dry_run)
        # ------------------------------------------------------------------
        batch_id = uuid.uuid4().hex[:12]
        now_utc = pd.Timestamp.utcnow()

        orders_df = pd.DataFrame(
            [
                {
                    "timestamp": now_utc,
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "exec_id": exec_id,
                    "strategy": strat_name,
                    "universe": universe,
                    "asof": asof,
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "notional": float(o["notional"]),
                    "trade_dollars": float(trade_dollars.get(o["symbol"], float("nan"))),
                    "equity": float(equity),
                    "dry_run": bool(dry_run),
                }
                for o in orders
            ]
        )

        orders_key = storage.write_orders_batch(
            strategy=strat_name,
            universe=universe,
            orders=orders_df,
            batch_id=batch_id,
        )
        logger.info(f"Saved planned orders | batch_id={batch_id} key={orders_key}")

        # ------------------------------------------------------------------
        # 5) Dry run or live execution
        # ------------------------------------------------------------------
        if dry_run:
            logger.warning(f"Dry run enabled — orders NOT sent | orders={orders}")
            return {
                "strategy_name": strat_name,
                "universe": universe,
                "asof": asof,
                "equity": equity,
                "target_weights": target_w.to_dict(),
                "orders": orders,
                "orders_key": orders_key,
                "batch_id": batch_id,
                "dry_run": True,
            }

        # Optional broker-side guard: open orders exist?
        if check_open_orders:
            # This requires your AlpacaTradingAPI to implement has_open_orders(symbol)
            symbols = sorted({o["symbol"] for o in orders})
            for sym in symbols:
                if hasattr(client, "has_open_orders") and client.has_open_orders(sym):  # type: ignore[attr-defined]
                    logger.warning(f"Skip (open orders exist) | symbol={sym}")
                    storage.write_last_exec(
                        strategy=strat_name,
                        universe=universe,
                        meta={
                            "asof": asof,
                            "status": "skipped_open_orders",
                            "batch_id": batch_id,
                            "run_id": run_id,
                            "exec_id": exec_id,
                            "timestamp": pd.Timestamp.utcnow().isoformat(),
                        },
                    )
                    return {
                        "strategy_name": strat_name,
                        "universe": universe,
                        "asof": asof,
                        "orders": orders,
                        "dry_run": False,
                        "reason": "open_orders_exist",
                        "batch_id": batch_id,
                        "orders_key": orders_key,
                    }

        # Mark submitted ASAP (idempotency on retries after partial submit)
        storage.write_last_exec(
            strategy=strat_name,
            universe=universe,
            meta={
                "asof": asof,
                "status": "submitted",
                "batch_id": batch_id,
                "run_id": run_id,
                "exec_id": exec_id,
                "submitted_at": pd.Timestamp.utcnow().isoformat(),
                "num_orders": len(orders),
                "equity": float(equity),
            },
        )

        # Execute sells first
        for o in [x for x in orders if x["side"] == "sell"]:
            logger.info(f"Submitting SELL | symbol={o['symbol']} notional={o['notional']:.2f}")
            client.place_order(symbol=o["symbol"], side="sell", notional=o["notional"])

        # Execute buys after
        for o in [x for x in orders if x["side"] == "buy"]:
            logger.info(f"Submitting BUY  | symbol={o['symbol']} notional={o['notional']:.2f}")
            client.place_order(symbol=o["symbol"], side="buy", notional=o["notional"])

        # ------------------------------------------------------------------
        # Save trades parquet (ledger)
        # ------------------------------------------------------------------
        trades_df = orders_df.copy()  # for MVP ledger mirrors planned orders
        trades_key = storage.write_trades_batch(
            strategy=strat_name,
            universe=universe,
            trades=trades_df,
            batch_id=batch_id,
        )
        logger.info(f"Saved trades parquet | batch_id={batch_id} key={trades_key}")

        # Mark completed
        storage.write_last_exec(
            strategy=strat_name,
            universe=universe,
            meta={
                "asof": asof,
                "status": "completed",
                "batch_id": batch_id,
                "run_id": run_id,
                "exec_id": exec_id,
                "completed_at": pd.Timestamp.utcnow().isoformat(),
                "num_orders": len(orders),
                "equity": float(equity),
                "orders_key": orders_key,
                "trades_key": trades_key,
            },
        )

        logger.info(f"Execution complete | strategy={strat_name} universe={universe} num_orders={len(orders)}")

        return {
            "strategy_name": strat_name,
            "universe": universe,
            "asof": asof,
            "equity": equity,
            "target_weights": target_w.to_dict(),
            "orders": orders,
            "batch_id": batch_id,
            "orders_key": orders_key,
            "trades_key": trades_key,
            "dry_run": False,
        }

    finally:
        # ------------------------------------------------------------------
        # Release lock
        # ------------------------------------------------------------------
        if lock_enabled and not dry_run:
            try:
                storage.clear_lock(strategy=strat_name, universe=universe)
            except Exception as e:
                logger.warning(f"Failed to clear lock | err={e!r}")

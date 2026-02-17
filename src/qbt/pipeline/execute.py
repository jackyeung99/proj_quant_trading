from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

import pandas as pd

from qbt.execution.alpaca_client import AlpacaTradingAPI
from qbt.execution.rebalancing import (
    latest_target_row,
    compute_target_dollars,
    compute_trade_dollars,
)
from qbt.core.logging import get_logger
from qbt.storage.artifacts import LiveStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Small helpers (pure functions) — keeps execute_weights readable
# ---------------------------------------------------------------------
def _read_knobs(execution_cfg: dict) -> dict:
    strat_name = execution_cfg.get("strategy_name")
    if not strat_name:
        raise ValueError("execution_cfg must include 'strategy_name'.")

    hp = execution_cfg.get("holding_period") or {}
    if not isinstance(hp, dict):
        hp = {}

    hp_tif = str(hp.get("tif", "opg")).lower()
    if hp_tif not in ("opg", "cls", "day"):
        raise ValueError(f"holding_period.tif must be one of 'opg','cls','day' (got {hp_tif!r})")

    return {
        # core
        "strat_name": strat_name,
        "universe": execution_cfg.get("universe", ""),
        "min_trade_dollars": float(execution_cfg.get("min_trade_dollars", 25.0)),
        "dry_run": bool(execution_cfg.get("dry_run", True)),
        # guards
        "lock_enabled": bool(execution_cfg.get("lock_enabled", True)),
        "allow_replay_same_asof": bool(execution_cfg.get("allow_replay_same_asof", False)),
        "check_open_orders": bool(execution_cfg.get("check_open_orders", True)),
        # ids
        "run_id": str(execution_cfg.get("run_id") or uuid.uuid4().hex[:12]),
        "exec_id": uuid.uuid4().hex[:12],
        # holding period
        "hp_enabled": bool(hp.get("enabled", False)),
        "hp_mode": str(hp.get("mode", "liquidate")).lower(),
        "hp_tif": hp_tif,
        "hp_symbols": hp.get("symbols", None),  # None or list[str]
        "hp_include_shorts": bool(hp.get("include_shorts", True)),
        "hp_skip_if_open_order": bool(hp.get("skip_if_open_order", True)),
        "hp_fail_on_errors": bool(hp.get("fail_on_errors", False)),
    }


def _acquire_lock(storage: LiveStore, *, strat: str, universe: str, meta: dict, enabled: bool, dry_run: bool) -> dict | None:
    if not enabled or dry_run:
        return None

    existing = storage.read_lock(strategy=strat, universe=universe)
    if existing.get("locked", False):
        return existing

    storage.write_lock(strategy=strat, universe=universe, meta=meta)
    return None


def _release_lock(storage: LiveStore, *, strat: str, universe: str, enabled: bool, dry_run: bool) -> None:
    if enabled and not dry_run:
        storage.clear_lock(strategy=strat, universe=universe)


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


def _load_target_weights(storage: LiveStore, *, strat: str, universe: str):
    weights_ts = storage.read_weights(strategy=strat, universe=universe)
    if weights_ts.empty:
        return None, None, None

    asof = str(weights_ts.index.max())
    target_row = latest_target_row(weights_ts)

    cash_like = {"CASH", "USD"}
    target_assets = [c for c in target_row.index if c not in cash_like]
    target_w = target_row.reindex(target_assets).fillna(0.0)

    return weights_ts, asof, target_w


def _build_orders_from_trade_dollars(trade_dollars: pd.Series) -> list[dict]:
    orders = []
    for sym, d in trade_dollars.items():
        side = "buy" if d > 0 else "sell"
        notional = round(float(abs(d)), 2)
        orders.append({"symbol": sym, "side": side, "notional": notional})
    return orders


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


def _skip_if_open_orders_exist(
    client: AlpacaTradingAPI,
    storage: LiveStore,
    *,
    strat: str,
    universe: str,
    asof: str,
    orders: list[dict],
    batch_id: str,
    run_id: str,
    exec_id: str,
    enabled: bool,
) -> dict | None:
    if not enabled:
        return None
    symbols = sorted({o["symbol"] for o in orders})
    for sym in symbols:
        if hasattr(client, "has_open_orders") and client.has_open_orders(sym):  # type: ignore[attr-defined]
            logger.warning(f"Skip (open orders exist) | symbol={sym}")
            storage.write_last_exec(
                strategy=strat,
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
            return {"symbol": sym, "reason": "open_orders_exist"}
    return None


def _submit_orders(client: AlpacaTradingAPI, orders: list[dict]) -> None:
    # sells first
    for o in (x for x in orders if x["side"] == "sell"):
        logger.info(f"Submitting SELL | symbol={o['symbol']} notional={o['notional']:.2f}")
        client.place_order(symbol=o["symbol"], side="sell", notional=o["notional"])
    # buys after
    for o in (x for x in orders if x["side"] == "buy"):
        logger.info(f"Submitting BUY  | symbol={o['symbol']} notional={o['notional']:.2f}")
        client.place_order(symbol=o["symbol"], side="buy", notional=o["notional"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def execute_weights(storage: LiveStore, execution_cfg: dict) -> dict:
    knobs = _read_knobs(execution_cfg)

    strat_name = knobs["strat_name"]
    universe = knobs["universe"]
    min_trade_dollars = knobs["min_trade_dollars"]
    dry_run = knobs["dry_run"]

    lock_enabled = knobs["lock_enabled"]
    allow_replay_same_asof = knobs["allow_replay_same_asof"]
    check_open_orders = knobs["check_open_orders"]

    run_id = knobs["run_id"]
    exec_id = knobs["exec_id"]

    hp_enabled = knobs["hp_enabled"]
    hp_mode = knobs["hp_mode"]
    hp_tif = knobs["hp_tif"]
    hp_symbols = knobs["hp_symbols"]
    hp_include_shorts = knobs["hp_include_shorts"]
    hp_skip_if_open_order = knobs["hp_skip_if_open_order"]
    hp_fail_on_errors = knobs["hp_fail_on_errors"]

    logger.info(
        f"Execution start | strategy={strat_name} universe={universe} dry_run={dry_run} "
        f"min_trade_dollars={min_trade_dollars:.2f} run_id={run_id} exec_id={exec_id} "
        f"holding_period_enabled={hp_enabled} holding_period_mode={hp_mode} holding_period_tif={hp_tif}"
    )

    lock = _acquire_lock(
        storage,
        strat=strat_name,
        universe=universe,
        enabled=lock_enabled,
        dry_run=dry_run,
        meta={
            "locked": True,
            "run_id": run_id,
            "exec_id": exec_id,
            "created_at": pd.Timestamp.utcnow().isoformat(),
        },
    )
    if lock is not None:
        logger.warning(f"Execution locked | strategy={strat_name} universe={universe} lock={lock}")
        return {
            "strategy_name": strat_name,
            "universe": universe,
            "orders": [],
            "dry_run": dry_run,
            "reason": "locked",
            "lock": lock,
        }

    try:
        client = AlpacaTradingAPI(cfg=execution_cfg.get("alpaca", {}) or {})
        logger.debug("Alpaca client initialized")

        # --------------------------------------------------------------
        # 0) Optional holding-period liquidation (before rebalance)
        # --------------------------------------------------------------
        hp_result: Optional[dict] = None
        if hp_enabled and hp_mode in ("liquidate", "liquidate_then_rebalance"):
            hp_result = _maybe_liquidate_for_holding_period(
                client,
                dry_run=dry_run,
                enabled=True,
                mode=hp_mode,
                tif=hp_tif,
                symbols=hp_symbols,
                include_shorts=hp_include_shorts,
                skip_if_open_order=hp_skip_if_open_order,
            )

            if hp_fail_on_errors and hp_result and hp_result.get("errors"):
                return {
                    "strategy_name": strat_name,
                    "universe": universe,
                    "orders": [],
                    "dry_run": dry_run,
                    "reason": "holding_period_liquidation_errors",
                    "holding_period": hp_result,
                }
            if hp_mode == "liquidate":
                return {
                    "strategy_name": strat_name,
                    "universe": universe,
                    "orders": [],
                    "dry_run": dry_run,
                    "reason": "holding_period_liquidation_only",
                    "holding_period": hp_result,
                }
                
                
        # --------------------------------------------------------------
        # 1) Load weights + idempotency
        # --------------------------------------------------------------
        weights_ts, asof, target_w = _load_target_weights(storage, strat=strat_name, universe=universe)

        if weights_ts is None or asof is None or target_w is None:
            logger.warning(f"No target weights found | strategy={strat_name} universe={universe}")
            return {
                "strategy_name": strat_name,
                "universe": universe,
                "orders": [],
                "dry_run": dry_run,
                "reason": "no_weights",
                "holding_period": hp_result,
            }

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
                "holding_period": hp_result,
            }

        nonzero_w = int((target_w.abs() > 1e-12).sum())
        logger.info(f"Loaded target weights | asof={asof} assets={len(target_w)} nonzero={nonzero_w}")

        # --------------------------------------------------------------
        # 2) Current state
        # --------------------------------------------------------------
        equity = float(client.get_equity())
        pos = client.get_active_positions()
        current_mv = {sym: float(p.market_value) for sym, p in pos.items()}
        current_dollars = pd.Series(current_mv, dtype=float).fillna(0.0)
        logger.info(f"Portfolio state | equity={equity:.2f} positions={len(current_dollars)}")

        # --------------------------------------------------------------
        # 3) Trade deltas
        # --------------------------------------------------------------
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
                "holding_period": hp_result,
            }

        logger.info(
            f"Trade deltas computed | n={int(trade_dollars.shape[0])} gross_notional={float(trade_dollars.abs().sum()):.2f}"
        )

        # --------------------------------------------------------------
        # 4) Orders + planned write
        # --------------------------------------------------------------
        orders = _build_orders_from_trade_dollars(trade_dollars)
        logger.info(f"Prepared orders | n={len(orders)}")

        batch_id, orders_key, orders_df = _write_planned_orders(
            storage,
            strat=strat_name,
            universe=universe,
            orders=orders,
            trade_dollars=trade_dollars,
            equity=equity,
            asof=asof,
            run_id=run_id,
            exec_id=exec_id,
            dry_run=dry_run,
            hp_enabled=hp_enabled,
            hp_mode=hp_mode,
            hp_tif=hp_tif,
        )
        logger.info(f"Saved planned orders | batch_id={batch_id} key={orders_key}")

        # --------------------------------------------------------------
        # 5) Dry run exit
        # --------------------------------------------------------------
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
                "holding_period": hp_result,
            }

        # --------------------------------------------------------------
        # 6) Broker-side open-order guard
        # --------------------------------------------------------------
        skip = _skip_if_open_orders_exist(
            client,
            storage,
            strat=strat_name,
            universe=universe,
            asof=asof,
            orders=orders,
            batch_id=batch_id,
            run_id=run_id,
            exec_id=exec_id,
            enabled=check_open_orders,
        )
        if skip is not None:
            return {
                "strategy_name": strat_name,
                "universe": universe,
                "asof": asof,
                "orders": orders,
                "dry_run": False,
                "reason": "open_orders_exist",
                "batch_id": batch_id,
                "orders_key": orders_key,
                "holding_period": hp_result,
                "skip": skip,
            }

        # --------------------------------------------------------------
        # 7) Mark submitted + submit orders
        # --------------------------------------------------------------
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
                "holding_period": hp_result or {},
            },
        )

        _submit_orders(client, orders)

        # --------------------------------------------------------------
        # 8) Optional holding-period liquidation (after rebalance)
        # --------------------------------------------------------------
        if hp_enabled and hp_mode == "rebalance_then_liquidate":
            hp_result = _maybe_liquidate_for_holding_period(
                client,
                dry_run=dry_run,
                enabled=True,
                mode=hp_mode,
                tif=hp_tif,
                symbols=hp_symbols,
                include_shorts=hp_include_shorts,
                skip_if_open_order=hp_skip_if_open_order,
            )
            if hp_fail_on_errors and hp_result and hp_result.get("errors"):
                logger.warning("Holding-period liquidation had errors after rebalance")

        # --------------------------------------------------------------
        # 9) Save trades + mark completed
        # --------------------------------------------------------------
        trades_df = orders_df.copy()
        trades_key = storage.write_trades_batch(
            strategy=strat_name,
            universe=universe,
            trades=trades_df,
            batch_id=batch_id,
        )
        logger.info(f"Saved trades parquet | batch_id={batch_id} key={trades_key}")

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
                "holding_period": hp_result or {},
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
            "holding_period": hp_result,
        }

    finally:
        try:
            _release_lock(storage, strat=strat_name, universe=universe, enabled=lock_enabled, dry_run=dry_run)
        except Exception as e:
            logger.warning(f"Failed to clear lock | err={e!r}")

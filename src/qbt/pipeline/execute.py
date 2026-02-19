from __future__ import annotations

from typing import Optional
import pandas as pd
import uuid 

from qbt.core.logging import get_logger
from qbt.execution.alpaca_client import AlpacaTradingAPI
from qbt.storage.artifacts import LiveStore

from qbt.execution.orders import _submit_orders 
from qbt.execution.guards import _skip_if_open_orders_exist, _acquire_lock, _release_lock

from qbt.execution.rebalancing import (
    plan_rebalance,
    _write_planned_orders, 
    _maybe_liquidate_for_holding_period, 
    _load_target_weights,
    _snapshot_portfolio,
    _load_prices
)


logger = get_logger(__name__)


def _ret_base(
    *,
    strat: str,
    universe: str,
    dry_run: bool,
    orders: list,
    reason: str | None = None,
    holding_period: dict | None = None,
    **extra,
) -> dict:
    out = {
        "strategy_name": strat,
        "universe": universe,
        "dry_run": bool(dry_run),
        "orders": orders,
    }
    if reason is not None:
        out["reason"] = reason
    if holding_period is not None:
        out["holding_period"] = holding_period
    out.update(extra)
    return out


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




def execute_weights(live_storage: LiveStore, execution_cfg: dict) -> dict:
    k = _read_knobs(execution_cfg)  # keep knobs grouped

    strat = k["strat_name"]
    universe = k["universe"]
    dry_run = k["dry_run"]

    logger.info(
        f"Execution start | strategy={strat} universe={universe} dry_run={dry_run} "
        f"min_trade_dollars={k['min_trade_dollars']:.2f} run_id={k['run_id']} exec_id={k['exec_id']} "
        f"holding_period_enabled={k['hp_enabled']} holding_period_mode={k['hp_mode']} holding_period_tif={k['hp_tif']}"
    )

    lock = _acquire_lock(
        live_storage,
        strat=strat,
        universe=universe,
        enabled=k["lock_enabled"],
        dry_run=dry_run,
        meta={
            "locked": True,
            "run_id": k["run_id"],
            "exec_id": k["exec_id"],
            "created_at": pd.Timestamp.utcnow().isoformat(),
        },
    )
    if lock is not None:
        logger.warning(f"Execution locked | strategy={strat} universe={universe} lock={lock}")
        return _ret_base(
            strat=strat,
            universe=universe,
            dry_run=dry_run,
            orders=[],
            reason="locked",
            lock=lock,
        )

    try:
        client = AlpacaTradingAPI(cfg=execution_cfg.get("alpaca", {}) or {})
        logger.debug("Alpaca client initialized")

        # --------------------------------------------------------------
        # 0) Optional holding-period liquidation (before rebalance)
        # --------------------------------------------------------------
        hp_result: Optional[dict] = None
        if k["hp_enabled"] and k["hp_mode"] in ("liquidate", "liquidate_then_rebalance"):
            hp_result = _maybe_liquidate_for_holding_period(
                client,
                dry_run=dry_run,
                enabled=True,
                mode=k["hp_mode"],
                tif=k["hp_tif"],
                symbols=k["hp_symbols"],
                include_shorts=k["hp_include_shorts"],
                skip_if_open_order=k["hp_skip_if_open_order"],
            )

            if k["hp_fail_on_errors"] and hp_result and hp_result.get("errors"):
                return _ret_base(
                    strat=strat,
                    universe=universe,
                    dry_run=dry_run,
                    orders=[],
                    reason="holding_period_liquidation_errors",
                    holding_period=hp_result,
                )

            if k["hp_mode"] == "liquidate":
                return _ret_base(
                    strat=strat,
                    universe=universe,
                    dry_run=dry_run,
                    orders=[],
                    reason="holding_period_liquidation_only",
                    holding_period=hp_result,
                )

        # --------------------------------------------------------------
        # 1) Load weights + idempotency
        # --------------------------------------------------------------
        _, asof, target_w = _load_target_weights(live_storage, strat=strat, universe=universe)

    

        if target_w is None or asof is None:
            logger.warning(f"No target weights found | strategy={strat} universe={universe}")
            return _ret_base(
                strat=strat,
                universe=universe,
                dry_run=dry_run,
                orders=[],
                reason="no_weights",
                holding_period=hp_result,
            )

        if (not k["allow_replay_same_asof"]) and live_storage.already_executed_asof(strategy=strat, universe=universe, asof=asof):
            last_exec = live_storage.read_last_exec(strategy=strat, universe=universe)
            logger.warning(f"Skip (already executed asof) | asof={asof} last_exec={last_exec}")
            return _ret_base(
                strat=strat,
                universe=universe,
                dry_run=dry_run,
                orders=[],
                reason="already_executed_asof",
                asof=asof,
                last_exec=last_exec,
                holding_period=hp_result,
            )

        nonzero_w = int((target_w.abs() > 1e-12).sum())
        logger.info(f"Loaded target weights | asof={asof} assets={len(target_w)} nonzero={nonzero_w}")

        # --------------------------------------------------------------
        # 2) Snapshot portfolio state
        # --------------------------------------------------------------
        equity, current_shares, current_dollars = _snapshot_portfolio(client)
        logger.info(f"Portfolio state | equity={equity:.2f} positions={len(current_dollars)}")

        # --------------------------------------------------------------
        # 3) Plan rebalance (ONE call)
        # --------------------------------------------------------------

        symbols = sorted(
            set(target_w.index.astype(str)) |
            set(current_shares.index.astype(str))
        )
    
        prices = _load_prices(client, symbols )


        plan = plan_rebalance(
            target_w=target_w,
            equity=equity,
            current_shares=current_shares,
            prices = prices,
            min_trade_dollars=k["min_trade_dollars"],
        )

        if not getattr(plan, "orders", None):
            logger.info(f"No trades required | strategy={strat} universe={universe}")
            return _ret_base(
                strat=strat,
                universe=universe,
                dry_run=dry_run,
                orders=[],
                reason="no_trades",
                asof=asof,
                equity=equity,
                target_weights=target_w.to_dict(),
                holding_period=hp_result,
            )


        logger.info(f"Plan built | n_orders={len(plan.orders)} gross_notional={float(plan.gross_notional):.2f}")

        # --------------------------------------------------------------
        # 4) Persist planned orders
        # --------------------------------------------------------------
        batch_id, orders_key, orders_df = _write_planned_orders(
            live_storage,
            strat=strat,
            universe=universe,
            plan=plan,
            equity=equity,
            asof=asof,
            run_id=k["run_id"],
            exec_id=k["exec_id"],
            dry_run=dry_run,
            hp_enabled=k["hp_enabled"],
            hp_mode=k["hp_mode"],
            hp_tif=k["hp_tif"],
        )
        logger.info(f"Saved planned orders | batch_id={batch_id} key={orders_key}")

        # --------------------------------------------------------------
        # 5) Dry run exit
        # --------------------------------------------------------------
        if dry_run:
            logger.warning("Dry run enabled — orders NOT sent")
            return _ret_base(
                strat=strat,
                universe=universe,
                dry_run=True,
                orders=plan.orders,
                asof=asof,
                equity=equity,
                target_weights=target_w.to_dict(),
                batch_id=batch_id,
                orders_key=orders_key,
                holding_period=hp_result,
            )

        # # --------------------------------------------------------------
        # # 6) Broker-side open-order guard
        # # --------------------------------------------------------------
        skip = _skip_if_open_orders_exist(
            client,
            live_storage,
            strat=strat,
            universe=universe,
            asof=asof,
            orders=plan.orders,
            batch_id=batch_id,
            run_id=k["run_id"],
            exec_id=k["exec_id"],
            enabled=k["check_open_orders"],
        )
        if skip is not None:
            return _ret_base(
                strat=strat,
                universe=universe,
                dry_run=False,
                orders=plan.orders,
                reason="open_orders_exist",
                asof=asof,
                batch_id=batch_id,
                orders_key=orders_key,
                holding_period=hp_result,
                skip=skip,
            )

        # --------------------------------------------------------------
        # 7) Mark submitted + submit orders
        # --------------------------------------------------------------
        live_storage.write_last_exec(
            strategy=strat,
            universe=universe,
            meta={
                "asof": asof,
                "status": "submitted",
                "batch_id": batch_id,
                "run_id": k["run_id"],
                "exec_id": k["exec_id"],
                "submitted_at": pd.Timestamp.utcnow().isoformat(),
                "num_orders": len(plan.orders),
                "equity": float(equity),
                "holding_period": hp_result or {},
            },
        )

        _submit_orders(client, plan.orders)

        # --------------------------------------------------------------
        # 8) Optional holding-period liquidation (after rebalance)
        # --------------------------------------------------------------
        if k["hp_enabled"] and k["hp_mode"] == "rebalance_then_liquidate":
            hp_result = _maybe_liquidate_for_holding_period(
                client,
                dry_run=dry_run,
                enabled=True,
                mode=k["hp_mode"],
                tif=k["hp_tif"],
                symbols=k["hp_symbols"],
                include_shorts=k["hp_include_shorts"],
                skip_if_open_order=k["hp_skip_if_open_order"],
            )
            if k["hp_fail_on_errors"] and hp_result and hp_result.get("errors"):
                logger.warning("Holding-period liquidation had errors after rebalance")

        # --------------------------------------------------------------
        # 9) Save trades + mark completed
        # --------------------------------------------------------------
        trades_df = orders_df.copy()
        trades_key = live_storage.write_trades_batch(
            strategy=strat,
            universe=universe,
            trades=trades_df,
            batch_id=batch_id,
        )
        logger.info(f"Saved trades parquet | batch_id={batch_id} key={trades_key}")

        live_storage.write_last_exec(
            strategy=strat,
            universe=universe,
            meta={
                "asof": asof,
                "status": "completed",
                "batch_id": batch_id,
                "run_id": k["run_id"],
                "exec_id": k["exec_id"],
                "completed_at": pd.Timestamp.utcnow().isoformat(),
                "num_orders": len(plan.orders),
                "equity": float(equity),
                "orders_key": orders_key,
                "trades_key": trades_key,
                "holding_period": hp_result or {},
            },
        )

        logger.info(f"Execution complete | strategy={strat} universe={universe} num_orders={len(plan.orders)}")

        return _ret_base(
            strat=strat,
            universe=universe,
            dry_run=False,
            orders=plan.orders,
            asof=asof,
            equity=equity,
            target_weights=target_w.to_dict(),
            batch_id=batch_id,
            orders_key=orders_key,
            trades_key=trades_key,
            holding_period=hp_result,
        )

    finally:
        try:
            _release_lock(live_storage, strat=strat, universe=universe, enabled=k["lock_enabled"], dry_run=dry_run)
        except Exception as e:
            logger.warning(f"Failed to clear lock | err={e!r}")

from __future__ import annotations

from typing import Optional, Any
import uuid

import pandas as pd

from qbt.core.logging import get_logger
from qbt.config.specs import StrategySpec
from qbt.storage.artifacts import LiveStore

from qbt.execution.orders import _submit_orders
from qbt.execution.guards import _skip_if_open_orders_exist, _acquire_lock, _release_lock
from qbt.execution.rebalancing import (
    plan_rebalance,
    _write_planned_orders,
    _maybe_liquidate_for_holding_period,
    _load_target_weights,
    _snapshot_portfolio,
    _load_prices,
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
    **extra: Any,
) -> dict[str, Any]:
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


def _read_execution_knobs(
    strategy: StrategySpec,
    *,
    run_id: str,
) -> dict[str, Any]:
    execution_cfg = strategy.execution or {}

    hp = execution_cfg.get("holding_period") or {}
    if not isinstance(hp, dict):
        hp = {}

    hp_tif = str(hp.get("tif", "opg")).lower()
    if hp_tif not in ("opg", "cls", "day"):
        raise ValueError(
            f"holding_period.tif must be one of 'opg','cls','day' (got {hp_tif!r})"
        )

    return {
        "strat_name": strategy.strategy_name,
        "universe": strategy.universe,
        "min_trade_dollars": float(execution_cfg.get("min_trade_dollars", 25.0)),
        "dry_run": bool(execution_cfg.get("dry_run", True)),
        "lock_enabled": bool(execution_cfg.get("lock_enabled", True)),
        "allow_replay_same_asof": bool(execution_cfg.get("allow_replay_same_asof", False)),
        "check_open_orders": bool(execution_cfg.get("check_open_orders", True)),
        "run_id": str(run_id),
        "exec_id": uuid.uuid4().hex[:12],
        "hp_enabled": bool(hp.get("enabled", False)),
        "hp_mode": str(hp.get("mode", "liquidate")).lower(),
        "hp_tif": hp_tif,
        "hp_symbols": hp.get("symbols", None),
        "hp_include_shorts": bool(hp.get("include_shorts", True)),
        "hp_skip_if_open_order": bool(hp.get("skip_if_open_order", True)),
        "hp_fail_on_errors": bool(hp.get("fail_on_errors", False)),
    }


def execute_weights(
    live_storage: LiveStore,
    strategy: StrategySpec,
    client: Any,
    *, 
    run_id: str,
) -> dict[str, Any]:
    k = _read_execution_knobs(strategy, run_id=run_id)

    strat = k["strat_name"]
    universe = k["universe"]
    dry_run = k["dry_run"]

    logger.info(
        "Execution start | strategy=%s universe=%s dry_run=%s "
        "min_trade_dollars=%.2f run_id=%s exec_id=%s "
        "holding_period_enabled=%s holding_period_mode=%s holding_period_tif=%s",
        strat,
        universe,
        dry_run,
        k["min_trade_dollars"],
        k["run_id"],
        k["exec_id"],
        k["hp_enabled"],
        k["hp_mode"],
        k["hp_tif"],
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
        logger.warning(
            "Execution locked | strategy=%s universe=%s lock=%s",
            strat,
            universe,
            lock,
        )
        return _ret_base(
            strat=strat,
            universe=universe,
            dry_run=dry_run,
            orders=[],
            reason="locked",
            lock=lock,
        )

    try:
        hp_result: Optional[dict[str, Any]] = None
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

        _, asof, target_w = _load_target_weights(
            live_storage,
            strat=strat,
            universe=universe,
        )

        if target_w is None or asof is None:
            logger.warning("No target weights found | strategy=%s universe=%s", strat, universe)
            return _ret_base(
                strat=strat,
                universe=universe,
                dry_run=dry_run,
                orders=[],
                reason="no_weights",
                holding_period=hp_result,
            )

        if (not k["allow_replay_same_asof"]) and live_storage.already_executed_asof(
            strategy=strat,
            universe=universe,
            asof=asof,
        ):
            last_exec = live_storage.read_last_exec(strategy=strat, universe=universe)
            logger.warning(
                "Skip (already executed asof) | asof=%s last_exec=%s",
                asof,
                last_exec,
            )
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
        logger.info(
            "Loaded target weights | asof=%s assets=%d nonzero=%d",
            asof,
            len(target_w),
            nonzero_w,
        )

        equity, current_shares, current_dollars = _snapshot_portfolio(client)
        logger.info(
            "Portfolio state | equity=%.2f positions=%d",
            equity,
            len(current_dollars),
        )

        symbols = sorted(
            set(target_w.index.astype(str)) |
            set(current_shares.index.astype(str))
        )
        prices = _load_prices(client, symbols)

        plan = plan_rebalance(
            target_w=target_w,
            equity=equity,
            current_shares=current_shares,
            prices=prices,
            min_trade_dollars=k["min_trade_dollars"],
        )

        if not getattr(plan, "orders", None):
            logger.info("No trades required | strategy=%s universe=%s", strat, universe)
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

        logger.info(
            "Plan built | n_orders=%d gross_notional=%.2f",
            len(plan.orders),
            float(plan.gross_notional),
        )

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
        logger.info("Saved planned orders | batch_id=%s key=%s", batch_id, orders_key)

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

        trades_df = orders_df.copy()
        trades_key = live_storage.write_trades_batch(
            strategy=strat,
            universe=universe,
            trades=trades_df,
            batch_id=batch_id,
        )
        logger.info("Saved trades parquet | batch_id=%s key=%s", batch_id, trades_key)

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

        logger.info(
            "Execution complete | strategy=%s universe=%s num_orders=%d",
            strat,
            universe,
            len(plan.orders),
        )

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
            _release_lock(
                live_storage,
                strat=strat,
                universe=universe,
                enabled=k["lock_enabled"],
                dry_run=dry_run,
            )
        except Exception as e:
            logger.warning("Failed to clear lock | err=%r", e)
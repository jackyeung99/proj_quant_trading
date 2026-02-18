from __future__ import annotations

from typing import Optional
import pandas as pd
import uuid 

from qbt.core.logging import get_logger
from qbt.execution.alpaca_client import AlpacaTradingAPI
from qbt.storage.artifacts import LiveStore


logger = get_logger(__name__)

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

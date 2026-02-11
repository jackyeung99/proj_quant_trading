
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Mapping, Any, Dict, Literal
from dataclasses import dataclass
from typing import Dict
import math
import requests

from qbt.core.types import Position



import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()

def _to_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TimeInForce = Literal["day", "gtc", "opg", "cls", "ioc", "fok"]
PositionIntent = Literal["buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"]


class AlpacaTradingAPI:


    def __init__(
        self, 
        cfg: Mapping[str, Any] | None = None,
        base_url: str = "https://paper-api.alpaca.markets/v2/",
        timeout_s: int = 60,
    ):

        self.cfg = cfg or {}

        self.base_url = base_url
        self.timeout_s = int(timeout_s)

        self.api_key = self.cfg.get(
            "api_key",
            os.getenv("ALPACA_API_KEY")
        )
        self.api_secret = self.cfg.get(
            "api_secret",
            os.getenv("ALPACA_API_SECRET")
        )

        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing Alpaca credentials (api_key/api_secret or env vars)")
        

    def get_equity(self) -> float:
        url = self.base_url + "account"
        headers = self._headers()

        r = requests.get(url, headers=headers, timeout=self.timeout_s)

        if r.status_code != 200:
            raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:300]}")

        js = r.json()

        # equity is a string, convert to float
        return float(js["equity"])



    def get_active_positions(self) -> Dict[str, Position]:
        """
        Returns positions keyed by symbol for easy downstream use:
            pos["XLE"].market_value, pos["XLE"].qty, etc.
        """
        url = self.base_url + "positions"
        headers = self._headers()

        r = requests.get(url, headers=headers, params={}, timeout=self.timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:300]}")

        js = r.json() or []
        out: Dict[str, Position] = {}

        for p in js:
            sym = p.get("symbol")
            if not sym:
                continue

            qty = _to_float(p.get("qty"))
            mv  = _to_float(p.get("market_value"))
            cp  = _to_float(p.get("current_price"))
            aep = _to_float(p.get("avg_entry_price"))
            cb  = _to_float(p.get("cost_basis"))
            upl = _to_float(p.get("unrealized_pl"))
            uplpc = _to_float(p.get("unrealized_plpc"))

            # Alpaca includes "side" on positions; if missing, infer from qty sign
            side = p.get("side") or ("short" if qty < 0 else "long")

            out[sym] = Position(
                symbol=sym,
                qty=qty,
                side=side,
                market_value=mv,
                current_price=cp,
                avg_entry_price=aep,
                cost_basis=cb,
                unrealized_pl=upl,
                unrealized_plpc=uplpc,
            )

        return out

    def place_order(
            self,
            *,
            symbol: str,
            side: Side,
            order_type: OrderType = "market",
            time_in_force: TimeInForce = "day",
            qty: Optional[float] = None,
            notional: Optional[float] = None,
            limit_price: Optional[float] = None,
            stop_price: Optional[float] = None,
            position_intent: Optional[PositionIntent] = None,
            client_order_id: Optional[str] = None,
        ) -> Dict[str, Any]:
        """
        Place an order via Alpaca.

        Notes:
        - Use POST /orders with JSON body.
        - Specify exactly one of qty or notional.
        - limit_price required for limit / stop_limit.
        - stop_price required for stop / stop_limit.

        Returns: Alpaca order JSON.
        """
        if not symbol:
            raise ValueError("symbol is required")

        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        if (qty is None) == (notional is None):
            raise ValueError("Provide exactly one of qty or notional")

        if qty is not None and qty <= 0:
            raise ValueError("qty must be > 0")

        if notional is not None and notional <= 0:
            raise ValueError("notional must be > 0")

        if order_type in ("limit", "stop_limit") and limit_price is None:
            raise ValueError("limit_price is required for limit/stop_limit orders")

        if order_type in ("stop", "stop_limit") and stop_price is None:
            raise ValueError("stop_price is required for stop/stop_limit orders")

        url = self.base_url + "orders"
        headers = self._headers()

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }

        # Alpaca expects strings for qty/price fields in some clients;
        # sending numbers usually works, but string is safest.
        if qty is not None:
            payload["qty"] = str(qty)
        if notional is not None:
            payload["notional"] = str(notional)

        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if stop_price is not None:
            payload["stop_price"] = str(stop_price)

        if position_intent is not None:
            payload["position_intent"] = position_intent

        if client_order_id is not None:
            payload["client_order_id"] = client_order_id

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)

        # Alpaca commonly returns 200 or 201 on success
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:300]}")

        return r.json()


    def _headers(self) -> dict:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing Alpaca API key/secret")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }
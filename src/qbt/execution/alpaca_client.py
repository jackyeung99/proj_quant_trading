
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Mapping, Any, Dict, Literal
from dataclasses import dataclass
from typing import Dict, List
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
        data_base_url: str = "https://data.alpaca.markets",
        timeout_s: int = 60,
    ):

        self.cfg = cfg or {}

        self.base_url = base_url
        self.data_base_url = data_base_url
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
        


    def get_historical_equity(
        self,
        *,
        period: str = "1Y",
        timeframe: str = "1D",
        extended_hours: bool = True,
        as_df: bool = True,
        tz: str = "UTC",
        date_start: str | None = None,   # "YYYY-MM-DD"
        date_end: str | None = None,     # "YYYY-MM-DD"
    ):
        url = self.base_url.rstrip("/") + "/account/portfolio/history"
        headers = self._headers()

        params = {
            "period": period,
            "timeframe": timeframe,
            "extended_hours": str(bool(extended_hours)).lower(),
        }
        if date_start:
            params["date_start"] = str(pd.Timestamp(date_start).date())
        if date_end:
            params["date_end"] = str(pd.Timestamp(date_end).date())

            r = requests.get(url, headers=headers, params=params, timeout=self.timeout_s)
            if r.status_code != 200:
                raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:300]}")

            js = r.json() or {}

        if not as_df:
            return js

        ts = js.get("timestamp") or []
        eq = js.get("equity") or []
        pl = js.get("profit_loss") or []
        plp = js.get("profit_loss_pct") or []

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(ts, unit="s", utc=True),
            "equity": pd.to_numeric(eq, errors="coerce"),
            "profit_loss": pd.to_numeric(pl, errors="coerce"),
            "profit_loss_pct": pd.to_numeric(plp, errors="coerce"),
        }).dropna(subset=["timestamp", "equity"]).sort_values("timestamp")

        df["period"] = period
        df["timeframe"] = timeframe
        df["extended_hours"] = bool(extended_hours)

        
        if tz and tz.upper() != "UTC":
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz)

        if "base_value" in js:
            df["base_value"] = _to_float(js.get("base_value"))

        return df
    
    def get_equity(self) -> float:
        url = self.base_url + "account"
        headers = self._headers()

        r = requests.get(url, headers=headers, timeout=self.timeout_s)

        if r.status_code != 200:
            raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:300]}")

        js = r.json()

        # equity is a string, convert to float
        return float(js["equity"])
    

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch latest prices from Alpaca Data API.

        Price selection priority:
        1) Mid-quote = (bid + ask) / 2
        2) Bid only
        3) Ask only
        4) Last trade price (fallback)
        5) Skip if no usable price

        Returns:
            dict[symbol -> price]
        """
        if not symbols:
            return {}

        uniq = list(dict.fromkeys([str(s).upper() for s in symbols]))

        headers = self._headers()

        # --------------------------------------------------
        # 1) Try latest QUOTES first
        # --------------------------------------------------
        url_q = f"{self.data_base_url}/v2/stocks/quotes/latest"
        r = requests.get(
            url_q,
            headers=headers,
            params={"symbols": ",".join(uniq)},
            timeout=self.timeout_s,
        )
        r.raise_for_status()

        data_q = r.json() or {}
        quotes = data_q.get("quotes") or {}

        out: Dict[str, float] = {}
        missing: List[str] = []

        for sym in uniq:
            q = quotes.get(sym)

            if not q:
                missing.append(sym)
                continue

            bid = _to_float(q.get("bp"), default=0.0)
            ask = _to_float(q.get("ap"), default=0.0)

            price = 0.0

            if bid > 0 and ask > 0:
                price = (bid + ask) / 2.0
            elif bid > 0:
                price = bid
            elif ask > 0:
                price = ask

            if price > 0:
                out[sym] = float(price)
            else:
                missing.append(sym)

        # --------------------------------------------------
        # 2) Fallback: latest TRADES for missing symbols
        # --------------------------------------------------
        if missing:
            url_t = f"{self.data_base_url}/v2/stocks/trades/latest"
            r = requests.get(
                url_t,
                headers=headers,
                params={"symbols": ",".join(missing)},
                timeout=self.timeout_s,
            )
            r.raise_for_status()

            data_t = r.json() or {}
            trades = data_t.get("trades") or {}

            for sym in missing:
                t = trades.get(sym)
                if not t:
                    continue

                price = _to_float(t.get("p"), default=0.0)
                if price > 0:
                    out[sym] = float(price)

        return out

    def get_latest_trade_prices(self, symbols: List[str]) -> Dict[str, float]:
        if not symbols:
            return {}

        uniq = list(dict.fromkeys([str(s).upper() for s in symbols]))
        url = f"{self.data_base_url}/v2/stocks/trades/latest"
        r = requests.get(url, headers=self._headers(), params={"symbols": ",".join(uniq)}, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json() or {}
        trades = data.get("trades") or {}

        out: Dict[str, float] = {}
        for sym in uniq:
            t = trades.get(sym)
            if not t:
                continue
            p = _to_float(t.get("p"), default=0.0)
            if p > 0:
                out[sym] = float(p)
        return out
    
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

    def has_open_orders(self, symbol: str | None = None) -> bool:
        orders = self.list_open_orders(symbol=symbol)  # implement using alpaca get_orders(status="open")
        return len(orders) > 0
    
    def list_open_orders(self, symbol: str | None = None):
        """
        GET /v2/orders?status=open[&symbols=XLE]
        Returns list[dict]
        """
        url = self.base_url + "orders"
        headers = self._headers()

        params = {"status": "open"}
        if symbol:
            # Alpaca expects 'symbols' (comma-separated) for filtering multiple.
            # For one symbol this is fine.
            params["symbols"] = symbol

        r = requests.get(url, headers=headers, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    

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

    def liquidate(
        self,
        *,
        symbol: str | None = None,
        at_open: bool = False,
        at_close: bool = False,
        skip_if_open_order: bool = True,
        include_shorts: bool = True,
        client_order_id_prefix: str = "liq",
    ) -> Dict[str, Any]:
        """
        Generic liquidation function.

        Parameters
        ----------
        symbol : optional str
            If provided, liquidate only this symbol.
            Otherwise liquidate all active positions.

        at_open : bool
            If True, submit Market-On-Open (opg).

        at_close : bool
            If True, submit Market-On-Close (cls).

        Default behavior (both False):
            Regular market order (day).

        Returns
        -------
        dict:
            {
              "submitted": [...],
              "skipped": [...],
              "errors": [...]
            }
        """

        if at_open and at_close:
            raise ValueError("Cannot set both at_open and at_close=True")

        if at_open:
            tif: TimeInForce = "opg"
        elif at_close:
            tif = "cls"
        else:
            tif = "day"

        positions = self.get_active_positions()

        if symbol:
            symbols = [symbol]
        else:
            symbols = list(positions.keys())

        submitted = []
        skipped = []
        errors = []

        for sym in symbols:
            pos = positions.get(sym)
            if not pos or _to_float(pos.qty) == 0:
                skipped.append({"symbol": sym, "reason": "no_position"})
                continue

            if skip_if_open_order:
                try:
                    if self.has_open_orders(symbol=sym):
                        skipped.append({"symbol": sym, "reason": "open_order_exists"})
                        continue
                except Exception as e:
                    skipped.append({"symbol": sym, "reason": f"open_order_check_failed: {e}"})
                    continue

            qty = abs(_to_float(pos.qty))
            if qty <= 0:
                skipped.append({"symbol": sym, "reason": "invalid_qty"})
                continue

            side = pos.side.lower()
            if side == "long":
                close_side: Side = "sell"
            elif side == "short":
                if not include_shorts:
                    skipped.append({"symbol": sym, "reason": "short_skipped"})
                    continue
                close_side = "buy"
            else:
                skipped.append({"symbol": sym, "reason": "unknown_side"})
                continue

            client_order_id = f"{client_order_id_prefix}_{sym}_{int(_time.time())}"[:48]

            try:
                order = self.place_order(
                    symbol=sym,
                    side=close_side,
                    order_type="market",
                    time_in_force=tif,
                    qty=qty,
                    position_intent=(
                        "sell_to_close" if close_side == "sell" else "buy_to_close"
                    ),
                    client_order_id=client_order_id,
                )
                submitted.append(order)
            except Exception as e:
                errors.append({"symbol": sym, "error": str(e)})

        return {
            "submitted": submitted,
            "skipped": skipped,
            "errors": errors,
        }

    def _headers(self) -> dict:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing Alpaca API key/secret")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }
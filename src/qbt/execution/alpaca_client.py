
from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Mapping, Any


import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()

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



    def get_active_positions(self):

        url = self.base_url  + 'positions'
        headers = self._headers()
        params = {}

        r = requests.get(url, headers=headers, params=params, timeout=self.timeout_s)
        
        if r.status_code != 200:
            raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:300]}")
        
        js = r.json()
        return js

    def place_order(self, 
                    symbol, 
                    qty, 
                    order_type, 
                    position_intent,
                    type = 'market',
                    time_in_force = 'gtc',
                    limit_price = None, 
                    stop_price = None,
    ): 
        
        url = self.base_url  + 'orders'
        headers = self._headers()

        params = {
            'symbol': symbol, 
            'qty': qty, 
            'side': order_type,
            'type': type,
            'time_in_force': time_in_force,
            'limit_price': limit_price,
            'stop_price': stop_price, 
            'position_intent': position_intent, 
    
        }

        r = requests.get(url, headers=headers, params=params, timeout=self.timeout_s)
        
        if r.status_code != 200:
            raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:300]}")
        
        js = r.json()
        pass
        


    def _headers(self) -> dict:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing Alpaca API key/secret")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }
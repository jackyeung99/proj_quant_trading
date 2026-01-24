# qbt/strategies/registry.py
from __future__ import annotations
from typing import Callable, Dict
from qbt.strategies.base import Strategy

_REGISTRY: Dict[str, Callable[[], Strategy]] = {}
_LOADED = False

def register_strategy(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def available_strategies() -> list[str]:
    ensure_strategies_loaded()
    return sorted(_REGISTRY.keys())

def create_strategy(name: str) -> Strategy:
    ensure_strategies_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]()

def ensure_strategies_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    # this import triggers qbt/strategies/__init__.py which imports all strategy modules
    import qbt.strategies  # noqa: F401
    _LOADED = True

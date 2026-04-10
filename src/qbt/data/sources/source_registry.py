# qbt/strategies/registry.py
from __future__ import annotations
from typing import Callable, Dict
from qbt.data.sources.source_base import DataSource

_REGISTRY: Dict[str, Callable[[], DataSource]] = {}
_LOADED = False

def source_kwargs_for_provider(provider: str, conn: dict) -> dict:
    if provider == "alpaca":
        return {
            "api_key": conn["api_key"],
            "api_secret": conn["api_secret"],
            "base_url": conn["data_base_url"],
            "feed": conn.get("feed", "iex"),
            "adjustment": conn.get("adjustment", "all"),
            "timeout_s": conn.get("timeout_s", 60),
            "limit": conn.get("limit", 10000),
            "sleep_s": conn.get("sleep_s", 0.25),
            "interval": conn.get("interval", "5Min"),
        }

    if provider == "fred":
        return {
            "api_key": conn["api_key"],
            "base_url": conn.get("base_url", "..."),
            "timeout_s": conn.get("timeout_s", 30),
        }

    raise ValueError(f"Unsupported provider {provider!r}")

def register_source(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def available_sources() -> list[str]:
    ensure_sources_loaded()
    return sorted(_REGISTRY.keys())

def create_source(name: str, cfg: dict) -> DataSource:
    """
    kwargs are passed to the source constructor (api_key, api_secret, etc.)
    """
    ensure_sources_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown source {name!r}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name](**cfg)

def ensure_sources_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    # this import triggers qbt/strategies/__init__.py which imports all strategy modules
    import qbt.data.sources  # noqa: F401
    _LOADED = True

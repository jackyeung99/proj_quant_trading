# qbt/strategies/registry.py
from __future__ import annotations
from typing import Callable, Dict
from qbt.data.sources.source_base import DataSource

_REGISTRY: Dict[str, Callable[[], DataSource]] = {}
_LOADED = False

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
    return _REGISTRY[name](cfg=cfg)

def ensure_sources_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    # this import triggers qbt/strategies/__init__.py which imports all strategy modules
    import qbt.data.sources  # noqa: F401
    _LOADED = True

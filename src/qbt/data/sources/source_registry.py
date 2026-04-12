from __future__ import annotations

from typing import Callable, Dict, Any

from qbt.data.sources.source_base import DataSource


_REGISTRY: Dict[str, Callable[..., DataSource]] = {}
_LOADED = False


def source_kwargs_for_provider(provider: str, conn: dict[str, Any]) -> dict[str, Any]:
    """
    Build constructor kwargs for a data source from resolved connection config.

    Important:
    - Only include long-lived constructor/configuration kwargs here
    - Request-time kwargs like feed, adjustment, interval, symbols, etc.
      should come from DatasetSpec and be passed to fetch(...)
    """
    if provider == "alpaca":
        return {
            "api_key": conn["api_key"],
            "api_secret": conn["api_secret"],
            "base_url": conn.get("base_url") or conn.get("data_base_url", "https://data.alpaca.markets"),
            "timeout_s": int(conn.get("timeout_s", 60)),
        }

    if provider == "fred":
        return {
            "api_key": conn["api_key"],
            "base_url": conn.get("base_url", "..."),
            "timeout_s": int(conn.get("timeout_s", 30)),
        }

    if provider == "yfinance":
        return {}

    if provider in {"parquet", "csv", "local_file"}:
        return {}

    raise ValueError(f"Unsupported provider {provider!r}")


def register_source(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def available_sources() -> list[str]:
    ensure_sources_loaded()
    return sorted(_REGISTRY.keys())


def create_source(name: str, cfg: dict[str, Any]) -> DataSource:
    """
    Instantiate a registered source using constructor kwargs.
    """
    ensure_sources_loaded()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown source {name!r}. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**cfg)


def ensure_sources_loaded() -> None:
    global _LOADED
    if _LOADED:
        return

    # importing qbt.data.sources triggers registration side effects
    import qbt.data.sources  # noqa: F401

    _LOADED = True
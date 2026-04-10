
import os
from typing import Any 

from qbt.execution.alpaca_client import AlpacaTradingClient
from qbt.data.sources.alpaca import AlpacaDataClient
from qbt.data.sources.source_registry import create_source, source_kwargs_for_provider

def resolve_connections(connections_cfg: dict) -> dict[str, dict]:
    resolved = {}

    for name, conn_cfg in connections_cfg.items():
    
        provider = conn_cfg["provider"]
        source = conn_cfg.get("credential_source", "none")
        endpoints = conn_cfg.get("endpoints", {})

        if source == "env":
            if provider == "alpaca":
                resolved[name] = {
                    "provider": provider,
                    "api_key": os.environ["ALPACA_API_KEY"],
                    "api_secret": os.environ["ALPACA_API_SECRET"],
                    **endpoints,
                }
       
            else:
                raise ValueError(f"Unsupported env-backed provider={provider}")

        elif source == "none":
            resolved[name] = {
                "provider": provider,
                **endpoints,
            }

        else:
            raise ValueError(f"Unsupported credential_source={source}")

    return resolved

def build_trading_clients(conn: dict):
    conn = conn['alpaca_trading']
    return {
        "trading_client": AlpacaTradingClient(
            api_key=conn["api_key"],
            api_secret=conn["api_secret"],
            base_url=conn["trading_base_url"],
        ),
    }


def required_data_source_refs(dataset_cfg: dict) -> set[str]:
    refs = set()

    for input_cfg in dataset_cfg.get("inputs", []):
        if not isinstance(input_cfg, dict):
            continue
        conn_name = input_cfg.get("connection")
        if conn_name:
            refs.add(conn_name)

    return refs


def build_data_clients(
    dataset_cfg: dict,
    resolved_connections: dict[str, dict],
) -> dict[str, Any]:
    built: dict[str, Any] = {}


    required_refs = required_data_source_refs(dataset_cfg)
    

    for conn_name in required_refs:
        if conn_name not in resolved_connections:
            raise KeyError(f"Missing resolved connection {conn_name!r}")

        conn = resolved_connections[conn_name]
        provider = conn["provider"]

        source_kwargs = {k: v for k, v in conn.items() if k != "provider"}
        kwargs = source_kwargs_for_provider(provider, conn)
  
        built[conn_name] = create_source(provider, kwargs)

    return built
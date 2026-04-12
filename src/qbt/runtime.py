from __future__ import annotations

import os
from typing import Any 

from qbt.execution.alpaca_client import AlpacaTradingClient
from qbt.storage.storage import make_storage
from qbt.storage.artifacts import LiveStore
from qbt.storage.paths import StoragePaths
from qbt.config.specs import ConnectionSpec, DatasetSpec, PipelineSpec, StrategySpec
from qbt.data.sources.source_registry import create_source, source_kwargs_for_provider

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunContext:
    run_id: str
    mode: str
    created_at_utc: str
    strategy_name: str
    strategy_class: str
    tag: str | None = None
    config_hash: str | None = None


@dataclass(frozen=True)
class ResolvedConnection:
    name: str
    provider: str
    credentials: dict[str, str] = field(default_factory=dict)
    endpoints: dict[str, str] = field(default_factory=dict)


@dataclass
class Runtime:
    storage: Any
    paths: Any
    artifact_store: Any

    data_providers: dict[str, Any] = field(default_factory=dict)
    brokers: dict[str, Any] = field(default_factory=dict)

    logger: Any = None
    run_ctx: RunContext | None = None




def resolve_connections(
    connection_specs: dict[str, ConnectionSpec],
) -> dict[str, ResolvedConnection]:
    resolved: dict[str, ResolvedConnection] = {}

    for name, spec in connection_specs.items():
        credentials: dict[str, str] = {}

        if spec.credential_source == "env":
            for logical_key, env_name in spec.env.items():
                value = os.getenv(env_name)
                if value is None:
                    raise ValueError(
                        f"Missing environment variable '{env_name}' "
                        f"for connection '{name}'"
                    )
                credentials[logical_key] = value

        elif spec.credential_source == "none":
            credentials = {}

        else:
            raise ValueError(
                f"Unsupported credential_source='{spec.credential_source}' "
                f"for connection '{name}'"
            )

        resolved[name] = ResolvedConnection(
            name=name,
            provider=spec.provider,
            credentials=credentials,
            endpoints=dict(spec.endpoints),
        )

    return resolved




def required_data_source_refs(dataset: DatasetSpec) -> set[str]:
    refs: set[str] = set()

    for source in dataset.sources:
        if source.connection:
            refs.add(source.connection)

    return refs


def build_brokers(
    resolved_connections: dict[str, ResolvedConnection],
) -> dict[str, Any]:
    brokers: dict[str, Any] = {}

    for name, conn in resolved_connections.items():
   
        if conn.name != "alpaca_trading":
            continue

        api_key = conn.credentials.get("api_key")
        api_secret = conn.credentials.get("api_secret")
        base_url = conn.endpoints.get("base_url") or conn.endpoints.get("trading_base_url")

        if not api_key or not api_secret:
            raise ValueError(
                f"Connection '{name}' is missing Alpaca trading credentials"
            )
        if not base_url:
            raise ValueError(
                f"Connection '{name}' is missing Alpaca trading base URL"
            )

        brokers[name] = AlpacaTradingClient(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
        )

    return brokers


def build_data_providers(
    dataset: DatasetSpec,
    resolved_connections: dict[str, ResolvedConnection],
) -> dict[str, Any]:
    built: dict[str, Any] = {}

    required_refs = required_data_source_refs(dataset)

    for conn_name in required_refs:
        if conn_name not in resolved_connections:
            raise KeyError(f"Missing resolved connection '{conn_name}'")

        conn = resolved_connections[conn_name]

        merged_cfg = {
            **conn.credentials,
            **conn.endpoints,
        }

        kwargs = source_kwargs_for_provider(conn.provider, merged_cfg)
        built[conn_name] = create_source(conn.provider, kwargs)

    return built

def build_run_context(
    *,
    pipeline: PipelineSpec,
    strategy: StrategySpec,
    run_id: str | None = None,
    config_hash: str | None = None,
) -> RunContext:

    created_at_utc = datetime.now(timezone.utc).isoformat()

    return RunContext(
        run_id=run_id,
        mode=pipeline.name,
        created_at_utc=created_at_utc,
        strategy_name=strategy.strategy_name,
        strategy_class=strategy.strategy_class,
        tag=strategy.tag,
        config_hash=config_hash,
    )

def build_runtime(
    *,
    pipeline: PipelineSpec,
    connections: dict[str, ConnectionSpec],
    dataset: DatasetSpec,
    strategy: StrategySpec,
    run_id: str,
    logger: Any = None,
) -> Runtime:
    storage = make_storage(pipeline.storage)
    paths = StoragePaths()
    artifact_store = LiveStore(storage, paths)

    resolved_connections = resolve_connections(connections)
    data_providers = build_data_providers(dataset, resolved_connections)
    brokers = build_brokers(resolved_connections)

    run_ctx = build_run_context(pipeline=pipeline, strategy=strategy, run_id=run_id)
    return Runtime(
        storage=storage,
        paths=paths,
        artifact_store=artifact_store,
        data_providers=data_providers,
        brokers=brokers,
        run_ctx=run_ctx,
    )
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


from qbt.pipeline.execute import execute_weights
from qbt.pipeline.evaluation import evaluate_portfolio
from qbt.pipeline.gold import build_gold_model_table
from qbt.pipeline.ingestion import ingest_all_sources
from qbt.pipeline.signal import signal
from qbt.pipeline.silver import canonicalize_all



@dataclass(frozen=True)
class PipelineDeps:
    storage: Any
    paths: Any
    artifact_store: Any
    data_sources: dict[str, Any] | None = None
    trading_clients: dict[str, Any] | None = None


@dataclass(frozen=True)
class PipelineContext:
    deps: PipelineDeps
    pipeline_cfg: dict
    dataset_cfg: dict
    model_table_cfg: dict
    strategy_cfg: dict
    deployment_cfg: dict
    connection_cfg: dict | None = None


@dataclass(frozen=True)
class PipelineStage:
    name: str
    runner: Callable[[PipelineContext], None]

    def enabled(self, pipeline_cfg: Mapping[str, Any]) -> bool:
        enabled = pipeline_cfg.get("enabled", {})
        return bool(enabled.get(self.name, False))


def _run_ingestion(ctx: PipelineContext) -> None:
    ingest_all_sources(
        ctx.deps.storage,
        ctx.deps.paths,
        dataset_cfg=ctx.dataset_cfg,
        sources=ctx.deps.data_sources or {},
    )


def _run_standardization(ctx: PipelineContext) -> None:
    canonicalize_all(
        ctx.deps.storage,
        ctx.deps.paths,
        dataset_cfg=ctx.dataset_cfg,
    )


def _run_gold(ctx: PipelineContext) -> None:
    build_gold_model_table(
        ctx.deps.storage,
        ctx.deps.paths,
        model_table_cfg=ctx.model_table_cfg,
    )


def _run_signal(ctx: PipelineContext) -> None:
    signal(
        live_storage=ctx.deps.artifact_store,
        strat_cfg=ctx.strategy_cfg,
        model_table_cfg=ctx.model_table_cfg,
    )


def _run_execution(ctx: PipelineContext) -> None:
    execute_weights(
        live_storage=ctx.deps.artifact_store,
        strategy_cfg=ctx.strategy_cfg,
        deployment_cfg=ctx.deployment_cfg,
        trading_client=getattr(ctx.deps.clients, "alpaca_trading", None) if ctx.deps.clients else None,
    )


def _run_evaluation(ctx: PipelineContext) -> None:
    evaluate_portfolio(
        live_storage=ctx.deps.artifact_store,
        strategy_cfg=ctx.strategy_cfg,
        deployment_cfg=ctx.deployment_cfg,
    )


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage("ingestion", _run_ingestion),
    PipelineStage("silver", _run_standardization),
    PipelineStage("gold", _run_gold),
    PipelineStage("signal", _run_signal),
    PipelineStage("execution", _run_execution),
    PipelineStage("evaluation", _run_evaluation),
)


def run_pipeline(
    *,
    storage: Any,
    paths: Any,
    cfg: dict,
    artifact_store: Any,
    data_sources : Any | None = None,
    trading_clients: Any | None = None
) -> None:
    ctx = PipelineContext(
        deps=PipelineDeps(
            storage=storage,
            paths=paths,
            artifact_store=artifact_store,
            data_sources=data_sources,
            trading_clients = trading_clients
        ),
        pipeline_cfg=cfg["pipeline"],
        dataset_cfg=cfg["dataset"],
        model_table_cfg=cfg["model_table"],
        strategy_cfg=cfg["strategy"],
        deployment_cfg=cfg["deployment"],
        connection_cfg=cfg.get("connection"),
    )

    for stage in PIPELINE_STAGES:
        if stage.enabled(ctx.pipeline_cfg):
            stage.runner(ctx)
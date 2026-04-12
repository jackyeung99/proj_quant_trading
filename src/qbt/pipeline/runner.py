from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from qbt.pipeline.execute import execute_weights
from qbt.pipeline.evaluation import evaluate_portfolio
from qbt.pipeline.gold import build_gold_model_table
from qbt.pipeline.ingestion import ingest_all_sources
from qbt.pipeline.signal import signal
from qbt.pipeline.standardization import canonicalize_all

from qbt.config.specs import PipelineSpec, DatasetSpec, StrategySpec
from qbt.runtime import Runtime


@dataclass
class PipelineState:
    table_ref: Any | None = None
    signal_ref: Any | None = None
    execution_result: Any | None = None
    evaluation_result: Any | None = None


@dataclass
class PipelineContext:
    runtime: Runtime
    pipeline: PipelineSpec
    dataset: DatasetSpec
    strategy: StrategySpec
    state: PipelineState = field(default_factory=PipelineState)


@dataclass(frozen=True)
class PipelineStage:
    name: str
    runner: Callable[[PipelineContext], None]

    def enabled(self, pipeline: PipelineSpec) -> bool:
        return bool(pipeline.enabled.get(self.name, False))


def _run_ingestion(ctx: PipelineContext) -> None:
    ingest_all_sources(
        storage=ctx.runtime.storage,
        paths=ctx.runtime.paths,
        dataset=ctx.dataset,
        sources=ctx.runtime.data_providers,
    )


def _run_standardization(ctx: PipelineContext) -> None:
    canonicalize_all(
        storage=ctx.runtime.storage,
        paths=ctx.runtime.paths,
        dataset=ctx.dataset,
    )


def _run_gold(ctx: PipelineContext) -> None:
    ctx.state.table_ref = build_gold_model_table(
        storage=ctx.runtime.storage,
        paths=ctx.runtime.paths,
        dataset=ctx.dataset,
    )


def _run_signal(ctx: PipelineContext) -> None:
    ctx.state.signal_ref = signal(
        live_storage=ctx.runtime.artifact_store,
        paths=ctx.runtime.paths,
        strategy=ctx.strategy
    )


def _run_execution(ctx: PipelineContext) -> None:
    conn_name = ctx.strategy.execution.get("connection")
    broker = ctx.runtime.brokers.get(conn_name)

    if broker is None:
        raise ValueError(
            f"No broker found for execution connection '{conn_name}'"
        )

    ctx.state.execution_result = execute_weights(
        live_storage=ctx.runtime.artifact_store,
        strategy=ctx.strategy,
        client=broker,
        run_id=ctx.runtime.run_ctx.run_id
    )


def _run_evaluation(ctx: PipelineContext) -> None:

    conn_name = ctx.strategy.execution.get("connection")
    broker = ctx.runtime.brokers.get(conn_name)

    ctx.state.evaluation_result = evaluate_portfolio(
        live_storage=ctx.runtime.artifact_store,
        strategy=ctx.strategy,
        client=broker,
    )


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage("ingestion", _run_ingestion),
    PipelineStage("standardization", _run_standardization),
    PipelineStage("feature_building", _run_gold),
    PipelineStage("signal_generation", _run_signal),
    PipelineStage("execution", _run_execution),
    PipelineStage("evaluation", _run_evaluation),
)


def run_pipeline(
    *,
    runtime: Runtime,
    pipeline: PipelineSpec,
    dataset: DatasetSpec,
    strategy: StrategySpec,
) -> None:
    ctx = PipelineContext(
        runtime=runtime,
        pipeline=pipeline,
        dataset=dataset,
        strategy=strategy,
    )

    for stage in PIPELINE_STAGES:

        if stage.enabled(ctx.pipeline):
            stage.runner(ctx)
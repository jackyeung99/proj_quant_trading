from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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


@dataclass(frozen=True)
class PipelineContext:
    deps: PipelineDeps
    cfg: dict
    sources_cfg: dict


@dataclass(frozen=True)
class PipelineStage:
    name: str
    runner: Callable[[PipelineContext, dict], None]

    def enabled(self, pipeline_cfg: dict) -> bool:
        return bool(pipeline_cfg.get(self.name, {}).get("enabled", False))

    def stage_cfg(self, pipeline_cfg: dict) -> dict:
        return pipeline_cfg.get(self.name, {}).get("cfg", {})


def _run_ingestion(ctx: PipelineContext, stage_cfg: dict) -> None:
    ingest_all_sources(
        ctx.deps.storage,
        ctx.deps.paths,
        ingestion_cfg=stage_cfg,
        sources_cfg=ctx.sources_cfg,
    )


def _run_silver(ctx: PipelineContext, stage_cfg: dict) -> None:
    canonicalize_all(
        ctx.deps.storage,
        ctx.deps.paths,
        sources_cfg=ctx.sources_cfg,
    )


def _run_gold(ctx: PipelineContext, stage_cfg: dict) -> None:
    build_gold_model_table(
        ctx.deps.storage,
        ctx.deps.paths,
        gold_cfg=stage_cfg,
    )


def _run_signal(ctx: PipelineContext, stage_cfg: dict) -> None:
    signal(
        live_storage=ctx.deps.artifact_store,
        strat_cfg=stage_cfg,
    )


def _run_execution(ctx: PipelineContext, stage_cfg: dict) -> None:
    execute_weights(
        live_storage=ctx.deps.artifact_store,
        execution_cfg=stage_cfg,
    )


def _run_evaluation(ctx: PipelineContext, stage_cfg: dict) -> None:
    evaluate_portfolio(
        live_storage=ctx.deps.artifact_store,
        execution_cfg=stage_cfg,
    )


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage("ingestion", _run_ingestion),
    PipelineStage("silver", _run_silver),
    PipelineStage("gold", _run_gold),
    PipelineStage("signal", _run_signal),
    PipelineStage("execution", _run_execution),
    PipelineStage("evaluation", _run_evaluation),
)


def run_pipeline(storage: Any, paths: Any, cfg: dict, artifact_store: Any) -> None:
    ctx = PipelineContext(
        deps=PipelineDeps(
            storage=storage,
            paths=paths,
            artifact_store=artifact_store,
        ),
        cfg=cfg,
        sources_cfg=cfg.get("sources", {}),
    )

    for stage in PIPELINE_STAGES:
        if stage.enabled(cfg):
            stage.runner(ctx, stage.stage_cfg(cfg))
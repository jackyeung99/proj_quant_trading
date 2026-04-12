from __future__ import annotations

from typing import Dict

from qbt.config.specs import (
    PipelineSpec,
    DatasetSpec,
    StrategySpec,
    ConnectionSpec,
)


def validate_specs(
    *,
    pipeline: PipelineSpec,
    dataset: DatasetSpec,
    strategy: StrategySpec,
    connections: Dict[str, ConnectionSpec],
) -> None:
    errors: list[str] = []

    # --------------------------------------------------
    # 1. Pipeline config sanity
    # --------------------------------------------------
    required_cfgs = {"dataset", "strategy", "connections"}
    missing_cfgs = required_cfgs - set(pipeline.configs.keys())

    if missing_cfgs:
        errors.append(
            f"Pipeline config missing required config references: {missing_cfgs}"
        )

    if not pipeline.stages:
        errors.append("Pipeline must define at least one stage")

    # --------------------------------------------------
    # 2. Stage consistency
    # --------------------------------------------------
    stage_set = set(pipeline.stages)
    enabled_set = set(pipeline.enabled.keys())

    unknown_enabled = enabled_set - stage_set
    if unknown_enabled:
        errors.append(
            f"'enabled' contains unknown stages: {unknown_enabled}"
        )

    missing_enabled = stage_set - enabled_set
    if missing_enabled:
        errors.append(
            f"Missing enabled flags for stages: {missing_enabled}"
        )

    # --------------------------------------------------
    # 3. Dataset validation
    # --------------------------------------------------
    if not dataset.sources:
        errors.append("Dataset must define at least one source")

    # Check all source connections exist
    for src in dataset.sources:
        if src.connection not in connections:
            errors.append(
                f"Dataset source '{src.name}' references unknown connection '{src.connection}'"
            )

    # --------------------------------------------------
    # 4. Strategy ↔ dataset validation
    # --------------------------------------------------
    if strategy.input_table:
        if strategy.input_table != dataset.table_name:
            errors.append(
                f"Strategy input_table '{strategy.input_table}' "
                f"does not match dataset.table_name '{dataset.table_name}'"
            )

    # --------------------------------------------------
    # 5. Execution validation
    # --------------------------------------------------
    execution_enabled = pipeline.enabled.get("execution", False)

    if execution_enabled:
        exec_cfg = strategy.execution

        if not exec_cfg:
            errors.append(
                "Execution stage enabled but strategy.execution is empty"
            )
        else:
            conn_name = exec_cfg.get("connection")

            if not conn_name:
                errors.append(
                    "Execution stage enabled but no connection specified in strategy.execution"
                )
            elif conn_name not in connections:
                errors.append(
                    f"Execution connection '{conn_name}' not found in connections config"
                )
            else:
                provider = connections[conn_name].provider
                # if "trading" not in provider:
                #     errors.append(
                #         f"Execution connection '{conn_name}' must be a trading provider, "
                #         f"got '{provider}'"
                #     )

    # --------------------------------------------------
    # 6. Data providers sanity
    # --------------------------------------------------
    for name, conn in connections.items():
        if not conn.provider:
            errors.append(f"Connection '{name}' has empty provider")

    # --------------------------------------------------
    # Final
    # --------------------------------------------------
    if errors:
        raise ValueError(
            "Config validation failed:\n\n" + "\n".join(f"- {e}" for e in errors)
        )
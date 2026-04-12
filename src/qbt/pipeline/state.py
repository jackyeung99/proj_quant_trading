from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TableRef:
    table_name: str
    path: str
    asof_utc: str | None = None
    row_count: int | None = None


@dataclass(frozen=True)
class SignalRef:
    signal_path: str
    weights_path: str | None = None
    generated_at_utc: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    orders_path: str | None = None
    fills_path: str | None = None
    submitted_at_utc: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    metrics_path: str | None = None
    equity_curve_path: str | None = None
    dashboard_merge_path: str | None = None


@dataclass
class PipelineState:
    table_ref: TableRef | None = None
    signal_ref: SignalRef | None = None
    execution_result: ExecutionResult | None = None
    evaluation_result: EvaluationResult | None = None
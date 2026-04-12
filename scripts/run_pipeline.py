from __future__ import annotations
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from qbt.config.loader import load_yaml
from qbt.config.parsers import (
    parse_pipeline_spec,
    parse_dataset_spec,
    parse_strategy_spec,
    parse_connection_specs,
)
from qbt.config.validation import validate_specs
from qbt.runtime import build_runtime
from qbt.pipeline.runner import run_pipeline
from qbt.utils.stamping import new_run_id
from qbt.core.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run QBT pipeline")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="configs/run.yaml",
        help="Path to pipeline YAML file",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    raw_pipeline = load_yaml(Path(args.pipeline))
    pipeline_spec = parse_pipeline_spec(raw_pipeline)


    raw_dataset = load_yaml(Path(pipeline_spec.configs["dataset"]))
    raw_strategy = load_yaml(Path(pipeline_spec.configs["strategy"]))
    raw_connections = load_yaml(Path(pipeline_spec.configs["connections"]))

    dataset_spec = parse_dataset_spec(raw_dataset)
    strategy_spec = parse_strategy_spec(raw_strategy)
    connection_specs = parse_connection_specs(raw_connections)

    validate_specs(
        pipeline=pipeline_spec,
        dataset=dataset_spec,
        strategy=strategy_spec,
        connections=connection_specs,
    )

    run_id = new_run_id()

    setup_logging(level=logging.INFO, log_file=None, force=True)

    runtime = build_runtime(
        pipeline=pipeline_spec,
        connections=connection_specs,
        dataset=dataset_spec,
        strategy=strategy_spec, 
        run_id=run_id,
    )


    run_pipeline(
        runtime=runtime,
        pipeline=pipeline_spec,
        dataset=dataset_spec,
        strategy=strategy_spec,
    )


if __name__ == "__main__":
    main()
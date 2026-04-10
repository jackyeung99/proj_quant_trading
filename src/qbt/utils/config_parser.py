import yaml
from pathlib import Path
from copy import deepcopy
import json
import hashlib
from typing import Any, Dict


def deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in {path}, got {type(data).__name__}")
    return data


def _resolve_path(raw_path: str, *, base_dir: Path) -> Path:
    p = Path(raw_path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def load_deployment_cfg(entry_path: str) -> dict[str, Any]:
    """
    Load a deployment config and all referenced configs into one resolved dict.

    Expected deployment shape:
      deployment: {...}
      refs:
        pipeline: "configs/pipeline/live_trading.yaml"
        dataset: "configs/datasets/sector_intraday_5m.yaml"
        model_table: "configs/model_tables/sector_state_signal.yaml"
        strategy: "configs/strategies/sector_long_only.yaml"
        connection: "configs/connections/alpaca_long_only.yaml"

    Optional inline overrides:
      overrides:
        strategy:
          params:
            gamma: 10
    """
    entry = Path(entry_path).resolve()
    root = _read_yaml(entry)
    base_dir = entry.parent.parent

    refs = root.get("refs", {})
    if not isinstance(refs, dict):
        raise ValueError(f"'refs' must be a mapping in {entry}")

    cfg: dict[str, Any] = {}

    # load referenced sections
    for section, raw_ref in refs.items():
        if not isinstance(raw_ref, str):
            raise ValueError(f"refs.{section} must be a string path, got {type(raw_ref).__name__}")

        ref_path = _resolve_path(raw_ref, base_dir=base_dir)
        loaded = _read_yaml(ref_path)

        # store the loaded config under the ref key, not merged into root flatly
        cfg[section] = loaded

    # copy deployment block directly
    deployment_block = root.get("deployment", {})
    if deployment_block:
        if not isinstance(deployment_block, dict):
            raise ValueError(f"'deployment' must be a mapping in {entry}")
        cfg["deployment"] = deployment_block

    # apply optional overrides onto loaded sections
    overrides = root.get("overrides", {})
    if overrides:
        if not isinstance(overrides, dict):
            raise ValueError(f"'overrides' must be a mapping in {entry}")
        for section, section_override in overrides.items():
            if not isinstance(section_override, dict):
                raise ValueError(
                    f"overrides.{section} must be a mapping, got {type(section_override).__name__}"
                )
            existing = cfg.get(section, {})
            if not isinstance(existing, dict):
                raise ValueError(f"Cannot apply dict override onto non-dict section '{section}'")
            cfg[section] = deep_merge(existing, section_override)

    return cfg
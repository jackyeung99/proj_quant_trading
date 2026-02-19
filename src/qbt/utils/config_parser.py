import yaml
from pathlib import Path
from copy import deepcopy
import json
import hashlib
from typing import Any, Dict


def config_hash(cfg: Dict[str, Any], *, length: int = 16) -> str:
    """
    Stable hash for a config dict.
    - order-independent
    - JSON-serializable
    - suitable for retrain invalidation
    """
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]

def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())

def deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out

def load_controlled_cfg(entry_path: str = "config/data.yaml") -> dict:
    entry = Path(entry_path).resolve()
    root = yaml.safe_load(entry.read_text()) or {}

    cfg: dict = {}
    config_dir = entry.parent  # root/config

    for section, control in root.items():

        if not isinstance(control, dict):
            cfg[section] = control
            continue

        config_path = control.get("config")

        # Only read enabled if user explicitly provided it
        has_enabled = "enabled" in control
        enabled = bool(control["enabled"]) if has_enabled else None

        loaded: dict = {}
        if config_path:
            p = Path(config_path)
            if not p.is_absolute():
                p = (config_dir / p).resolve()

            if not p.exists():
                raise FileNotFoundError(f"[{section}] config file not found: {p}")

            loaded = yaml.safe_load(p.read_text()) or {}

        # merge: file contents + inline overrides (except "config" and "enabled")
        inline_overrides = {k: v for k, v in control.items() if k not in ("enabled", "config")}

        merged = deep_merge(loaded, inline_overrides)

        # For pipeline steps, keep enabled separate from the cfg payload
        if section in {"ingestion", "silver", "gold", "signal", "execution", "evaluation"}:
            cfg[section] = {"enabled": enabled, "cfg": merged}
        else:
            # non-step sections like storage/sources are always "enabled"
            cfg[section] = merged

    return cfg
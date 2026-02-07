import yaml
from pathlib import Path
from copy import deepcopy

def deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out

def load_controlled_cfg(entry_path: str = "config/data.yaml") -> dict:
    """
    Assumptions:
      - you run from repo root
      - entry file is in root/config/data.yaml
      - each section in data.yaml has:
            enabled: bool
            config: "storage.yaml"  (relative to the config/ folder) OR an absolute path
      - inline overrides in data.yaml are merged on top of the loaded file
    print(entry_path)
    """
    entry = Path(entry_path).resolve()
    root = yaml.safe_load(entry.read_text()) or {}


    cfg: dict = {}
    config_dir = entry.parent  # root/config


    for section, control in root.items():
        

        if not isinstance(control, dict):
            cfg[section] = control
            continue

        enabled = bool(control.get("enabled", False))
        config_path = control.get("config")

        loaded: dict = {}
        if config_path:
            p = Path(config_path)
            if not p.is_absolute():
                # resolve relative to config/ directory
                p = (config_dir / p).resolve()

            if not p.exists():
                raise FileNotFoundError(f"[{section}] config file not found: {p}")

            loaded = yaml.safe_load(p.read_text()) or {}

        # merge: file contents + inline overrides (except "config")
        overrides = {k: v for k, v in control.items() if k != "config"}
        merged = deep_merge(loaded, overrides)
        merged["enabled"] = enabled
        cfg[section] = merged

    return cfg



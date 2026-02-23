import pandas as pd 
import json
import hashlib
from typing import Any, Dict

def make_snapshot_id(trained_at_utc: str, config_hash: str | None = None) -> str:
    ts = pd.to_datetime(trained_at_utc, utc=True, errors="coerce")
    if ts is pd.NaT:
        base = "unknown"
    else:
        base = ts.strftime("%Y%m%dT%H%M%SZ")
    if config_hash:
        base = f"{base}_{str(config_hash)[:7]}"
    return base


def config_hash(cfg: Dict[str, Any], *, length: int = 16) -> str:
    """
    Stable hash for a config dict.
    - order-independent
    - JSON-serializable
    - suitable for retrain invalidation
    """
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
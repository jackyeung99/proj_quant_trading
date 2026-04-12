from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from qbt.config.specs import DatasetSpec
from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths


def _stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def write_gold_wide_with_manifest(
    storage: Storage,
    paths: StoragePaths,
    df: pd.DataFrame,
    *,
    dataset: DatasetSpec,
) -> dict[str, Any]:
    """
    Persist wide gold table + manifest.

    Expected df shape:
      - index is session_date, or
      - session_date exists as a column

    Wide means:
      - no 'ticker' column
      - feature columns are already pivoted wide
    """
    if df is None or df.empty:
        return {"written": False, "reason": "empty_df"}

    out = df.copy()

    # normalize session_date
    if "session_date" in out.columns:
        out["session_date"] = pd.to_datetime(out["session_date"], utc=True, errors="coerce")
        out = out.dropna(subset=["session_date"])
        out = out.sort_values("session_date")
        out = out.set_index("session_date")
    else:
        idx = pd.to_datetime(out.index, utc=True, errors="coerce")
        out.index = idx
        out.index.name = "session_date"
        out = out[~out.index.isna()].sort_index()

    if "ticker" in out.columns:
        raise ValueError("Wide gold table should not have a 'ticker' column. Pivot before writing.")

    def _col_name(c: Any) -> str:
        if isinstance(c, tuple):
            return "|".join(str(x) for x in c)
        return str(c)

    feature_cols = [_col_name(c) for c in out.columns]

    aggregation = dict(dataset.aggregation or {})
    features = dict(dataset.features or {})
    output = dict(dataset.output or {})

    cfg_snapshot = {
        "dataset": {
            "name": dataset.name,
            "mode": dataset.mode,
            "table_name": dataset.table_name,
            "assets": dataset.assets,
        },
        "aggregation": {
            "input_freq": aggregation.get("input_freq"),
            "agg_freq": aggregation.get("agg_freq"),
            "market_tz": aggregation.get("market_tz"),
            "cutoff_hour": aggregation.get("cutoff_hour"),
            "daily_assets": aggregation.get("daily_assets", []),
        },
        "features": {
            "intraday": features.get("intraday", []),
            "daily_transforms": features.get("daily_transforms", []),
        },
        "output": output,
    }
    spec_hash = _stable_hash(cfg_snapshot)

    dt_min = out.index.min()
    dt_max = out.index.max()

    cols_meta = [{"name": _col_name(c), "dtype": str(out[c].dtype)} for c in out.columns]

    freq = aggregation.get("agg_freq", "1D")
    tag = dataset.table_name

    table_key = paths.gold_table_key(freq=freq, tag=tag)
    storage.write_parquet(out.reset_index(), table_key)

    manifest_key = paths.gold_manifest_key(freq=freq, tag=tag)
    manifest = {
        "dataset": dataset.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freq": freq,
        "table_name": tag,
        "spec_hash": spec_hash,
        "spec": cfg_snapshot,
        "shape": {
            "rows": int(out.shape[0]),
            "cols": int(out.shape[1]),
            "start_utc": None if pd.isna(dt_min) else dt_min.isoformat(),
            "end_utc": None if pd.isna(dt_max) else dt_max.isoformat(),
        },
        "schema": {
            "index": {"name": "session_date", "dtype": str(out.index.dtype)},
            "feature_columns": feature_cols,
            "columns": cols_meta,
        },
        "asof_convention": (
            "Row at session_date=t contains fields/features computed using data available up to the cutoff for t. "
            "Execution shifting should happen in portfolio simulation (weights shifted to t+1)."
        ),
        "paths": {
            "table_parquet": table_key,
            "manifest_json": manifest_key,
        },
    }

    storage.write_json(manifest, manifest_key)

    return {
        "written": True,
        "spec_hash": spec_hash,
        "table_key": table_key,
        "manifest_key": manifest_key,
    }
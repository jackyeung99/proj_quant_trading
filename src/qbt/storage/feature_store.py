from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from typing import Any
import pandas as pd

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
    gold_cfg: dict,
) -> dict[str, Any]:
    """
    Persist WIDE gold table + manifest.

    Expected df shape (wide):
      - index: session_date (datetime-like)
      - columns: features (and optionally a second level for ticker)
        OR
      - columns include 'session_date' and features, with no 'ticker' column

    Writes:
      - gold/freq=<freq>/tag=<tag>/table.parquet
      - gold/freq=<freq>/tag=<tag>/_manifest.json
    """
    if df is None or df.empty:
        return {"written": False, "reason": "empty_df"}

    out = df.copy()

    # ---- normalize session_date ----
    if "session_date" in out.columns:
        out["session_date"] = pd.to_datetime(out["session_date"], utc=True, errors="coerce")
        out = out.dropna(subset=["session_date"])
        out = out.sort_values("session_date")
        out = out.set_index("session_date")
    else:
        # assume index is session_date
        idx = pd.to_datetime(out.index, utc=True, errors="coerce")
        out.index = idx
        out.index.name = "session_date"
        out = out[~out.index.isna()].sort_index()

    # ---- WIDE schema: no ticker column ----
    if "ticker" in out.columns:
        raise ValueError("Wide gold table should not have a 'ticker' column. Pivot before writing.")

    # feature columns (handles MultiIndex columns too)
    def _col_name(c: Any) -> str:
        if isinstance(c, tuple):
            return "|".join(str(x) for x in c)
        return str(c)

    feature_cols = [_col_name(c) for c in out.columns]

    # ---- cfg snapshot + hash ----
    cfg_snapshot = {
        "gold": {
            "input_freq": gold_cfg.get("input_freq"),
            "agg_freq": gold_cfg.get("agg_freq"),
            "market_tz": gold_cfg.get("market_tz"),
            "cutoff_hour": gold_cfg.get("cutoff_hour"),
            "intraday_assets": gold_cfg.get("daily_assets", []),
            "daily_assets": gold_cfg.get("daily_assets", []),
            "intra_features": gold_cfg.get("intra_features", []),
            "daily_transforms": gold_cfg.get("daily_transforms", []),
        }
    }
    spec_hash = _stable_hash(cfg_snapshot)

    # ---- dataset stats ----
    dt_min = out.index.min()
    dt_max = out.index.max()

    # schema metadata
    cols_meta = [{"name": _col_name(c), "dtype": str(out[c].dtype)} for c in out.columns]

    # ---- write parquet ----
    freq = gold_cfg.get("agg_freq")
    tag = gold_cfg.get("tag")
    table_key = paths.gold_table_key(freq=freq, tag=tag)
    storage.write_parquet(out.reset_index(), table_key)  # store session_date as a column

    # ---- write manifest json ----
    manifest_key = paths.gold_manifest_key(freq=freq, tag=tag)
    manifest = {
        "dataset": "gold_wide_table",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freq": freq,
        "tag": tag,
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
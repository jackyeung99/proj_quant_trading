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


def write_gold_long_with_manifest(
    storage: Storage,
    paths: StoragePaths,
    df: pd.DataFrame,
    *,
    gold_cfg: dict,
) -> dict:
    """
    Persist LONG gold table + manifest.

    Writes:
      - gold/freq=<freq>/tag=<tag>/table.parquet
      - gold/freq=<freq>/tag=<tag>/_manifest.json
    """
    if df is None or df.empty:
        return {"written": False, "reason": "empty_df"}

    # --- validate + normalize ---
    required = {"session_date", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gold df missing required columns: {sorted(missing)}")

    out = df.copy()
    out["session_date"] = pd.to_datetime(out["session_date"], utc=True, errors="coerce")
    out["ticker"] = out["ticker"].astype("string")

    out = out.dropna(subset=["session_date", "ticker"]).sort_values(["ticker", "session_date"]).reset_index(drop=True)

    # --- feature columns ---
    feature_cols = [c for c in out.columns if c not in ("session_date", "ticker")]

    # --- cfg snapshot + hash (what produced this dataset) ---
    cfg_snapshot = {
        "gold": {
            "input_freq": gold_cfg.get("input_freq"),
            "agg_freq": gold_cfg.get("agg_freq"),
            "market_tz": gold_cfg.get("market_tz"),
            "cutoff_hour": gold_cfg.get("cutoff_hour"),
            "assets": gold_cfg.get("assets", []),
            "intra_features": gold_cfg.get("intra_features", []),
            "daily_transforms": gold_cfg.get("daily_transforms", []),
        }
    }
    spec_hash = _stable_hash(cfg_snapshot)

    # --- compute dataset stats ---
    tickers = sorted(out["ticker"].dropna().unique().tolist())
    dt_min = out["session_date"].min()
    dt_max = out["session_date"].max()

    # per-feature dtype summary
    cols_meta = [{"name": str(c), "dtype": str(out[c].dtype)} for c in out.columns]

    # --- write parquet ---
    freq = gold_cfg.get("agg_freq")
    tag = gold_cfg.get("tag")
    table_key = paths.gold_table_key(freq=freq, tag=tag)
    storage.write_parquet(out, table_key)

    # --- write manifest json ---
    manifest = {
        "dataset": "gold_long_table",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freq": freq,
        "tag": tag,
        "spec_hash": spec_hash,
        "spec": cfg_snapshot,
        "shape": {
            "rows": int(out.shape[0]),
            "cols": int(out.shape[1]),
            "n_assets": int(len(tickers)),
            "start_utc": None if pd.isna(dt_min) else dt_min.isoformat(),
            "end_utc": None if pd.isna(dt_max) else dt_max.isoformat(),
        },
        "assets": tickers,
        "schema": {
            "required_columns": ["datetime", "ticker"],
            "feature_columns": feature_cols,
            "columns": cols_meta,
        },
        "asof_convention": (
            "Row at datetime=t contains fields/features computed using data available up to t. "
            "Execution shifting should happen in portfolio simulation (weights shifted to t+1)."
        ),
        "paths": {
            "table_parquet": table_key,
            "manifest_json": paths.gold_manifest_key(freq=freq, tag=tag),
        },
    }

    manifest_key = paths.gold_manifest_key(freq=freq, tag=tag)
    storage.write_json(manifest, manifest_key)  # <-- implement write_text or use write_json

    return {"written": True, "spec_hash": spec_hash, "table_key": table_key, "manifest_key": manifest_key}

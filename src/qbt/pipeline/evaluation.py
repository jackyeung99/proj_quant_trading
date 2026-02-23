from __future__ import annotations

from typing import Any, Dict, Iterable
import pandas as pd

from qbt.core.logging import get_logger
from qbt.execution.alpaca_client import AlpacaTradingAPI
from qbt.storage.artifacts import LiveStore
from qbt.metrics.summary import compute_metrics_simple
from qbt.data.merge import join_daily, fill_in_force

# NOTE: consider moving these into qbt.data.merge / qbt.utils.dates for reuse
from qbt.utils.dates import _ensure_session_index


logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Defaults / configuration
# ---------------------------------------------------------------------

DEFAULT_STAMP_COLS: set[str] = {
    # join key (sometimes stored as column)
    "session_date",
    # common stamps
    "asof_utc",
    "generated_at_utc",
    "trained_at_utc",
    "trained_at",
    "train_start_ts_utc",
    "train_end_ts_utc",
    # lineage / ids
    "config_hash",
    "snapshot_id",
    "model_snapshot_id",
    # misc
    "market_tz",
    "cutoff_hour",
    "period",
    "timeframe",
    "extended_hours",
    "base_value",
}


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _equity_to_returns(df: pd.DataFrame, *, equity_col: str = "portfolio_value") -> pd.Series:
    eq = pd.to_numeric(df[equity_col], errors="coerce").astype(float)
    return eq.pct_change().fillna(0.0)


def prep_state_ts(
    df: pd.DataFrame | None,
    *,
    keep_cols: list[str] | None = None,
    drop_cols: set[str] | None = None,
    rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Generic prep for any time-series artifact indexed (or indexable) by session_date.
    - normalizes index using _ensure_session_index
    - optional rename
    - optional keep/drop columns
    - sorts + de-dupes index (keep last)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    x = _ensure_session_index(df).sort_index()

    if rename:
        x = x.rename(columns=rename)

    if keep_cols is not None:
        cols = [c for c in keep_cols if c in x.columns]
        x = x[cols]
    elif drop_cols:
        x = x[[c for c in x.columns if c not in drop_cols]]

    # keep last snapshot for that session_date
    x = x[~x.index.duplicated(keep="last")]
    return x


def prep_weights_wide(
    weights: pd.DataFrame | None,
    *,
    stamp_cols: set[str] = DEFAULT_STAMP_COLS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert weights snapshots into a wide daily table:
      index = session_date
      cols  = <asset>_weight

    Any columns that look like stamps/lineage are ignored via stamp_cols.
    """
    w = prep_state_ts(weights)
    if w.empty:
        return pd.DataFrame(), []

    asset_cols = [c for c in w.columns if c not in stamp_cols]

    # numeric + fill
    for c in asset_cols:
        w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0)

    rename_map = {c: f"{c}_weight" for c in asset_cols}
    w = w.rename(columns=rename_map)

    weight_cols = list(rename_map.values())
    w = w[weight_cols].astype(float)

    return w, weight_cols


def prep_meta_ts(meta: pd.DataFrame | None) -> pd.DataFrame:
    """
    Meta is already "meta columns"; we only ensure session index + dedupe.
    You may drop large nested objects here if present.
    """
    # Example: drop large nested dict column if exists
    drop = set()
    if meta is not None and not meta.empty and "strategy_meta" in meta.columns:
        drop.add("strategy_meta")
    return prep_state_ts(meta, drop_cols=drop if drop else None)


def prep_gold_wide(
    gold: pd.DataFrame | None,
    *,
    session_col: str = "session_date",
    ticker_col: str = "ticker",
    drop_cols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Convert long gold table into wide format:
        index = session_date
        columns = <ticker>_<feature>
    """
    if gold is None or gold.empty:
        return pd.DataFrame()

    df = gold.copy()

    df[session_col] = pd.to_datetime(df[session_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[session_col, ticker_col])

    id_cols = {session_col, ticker_col}
    if drop_cols:
        id_cols |= set(drop_cols)

    value_cols = [c for c in df.columns if c not in id_cols]

    wide = (
        df.pivot_table(
            index=session_col,
            columns=ticker_col,
            values=value_cols,
            aggfunc="last",
        )
        .sort_index()
    )

    if isinstance(wide.columns, pd.MultiIndex):
        # (feature, ticker) -> ticker_feature
        wide.columns = [f"{t}_{f}" for f, t in wide.columns]

    wide.index.name = session_col
    return wide


def prep_equity_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Alpaca portfolio history to:
      index = session_date (as provided by caller)
      col   = portfolio_value
    """
    x = df.copy()

    if "equity" in x.columns and "portfolio_value" not in x.columns:
        x = x.rename(columns={"equity": "portfolio_value"})

    # Expect caller already provides a daily timestamp column; we treat it as session_date.
    x = (
        x.rename(columns={"timestamp": "session_date"})
         .set_index("session_date")
         .sort_index()
    )

    x["portfolio_value"] = pd.to_numeric(x["portfolio_value"], errors="coerce")

    # Drop rows before first non-zero equity (avoid long zero warmup)
    mask_started = x["portfolio_value"] != 0.0
    if mask_started.any():
        first_valid = mask_started.idxmax()
        x = x.loc[first_valid:]
    else:
        return x.iloc[0:0]

    return x[["portfolio_value"]]



# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------

def evaluate_portfolio(live_storage: LiveStore, execution_cfg: dict) -> dict:
    """
    Evaluation step:
      1) fetch portfolio equity history (UTC)
      2) load weights snapshots, model meta snapshots, gold features
      3) canonicalize & merge on session_date
      4) in-force forward-fill
      5) compute returns & metrics
      6) persist panel + metrics
    """
    client = AlpacaTradingAPI(cfg=execution_cfg.get("alpaca", {}) or {})
    logger.debug("Alpaca client initialized")

    strategy = execution_cfg.get("strategy_name")
    universe = execution_cfg.get("universe")
    gold_path = execution_cfg["gold_path"]

    # ------------------------------------------------------------
    # 1) Portfolio history (Alpaca)
    # ------------------------------------------------------------
    end_utc = pd.Timestamp.now(tz="UTC")
    end_iso = end_utc.isoformat()

    port_cfg = execution_cfg.get("portfolio", {}) or {}
    period = port_cfg.get("period", "2M")
    timeframe = port_cfg.get("timeframe", "1D")

    portfolio_raw = client.get_historical_equity(
        period=period,
        timeframe=timeframe,
        extended_hours=False,
        as_df=True,
        tz="UTC",
        date_end=end_iso,
    )

    # If Alpaca returns UTC timestamps at midnight, your session_date should be market day label.
    # Ideally: convert portfolio_raw["timestamp"] from UTC -> market_tz then normalize.
    market_tz = execution_cfg.get("market_tz", "America/New_York")
    ts_utc = pd.to_datetime(portfolio_raw["timestamp"], utc=True, errors="coerce")
    portfolio_raw = portfolio_raw.copy()
    portfolio_raw["timestamp"] = ts_utc.dt.tz_convert(market_tz).dt.normalize().dt.tz_localize(None)

    portfolio = prep_equity_panel(portfolio_raw)
    # print(portfolio)

    # ------------------------------------------------------------
    # 2) Weights snapshots
    # ------------------------------------------------------------
    weights_raw = live_storage.read_all_weights(strategy=strategy, universe=universe)
    weights_wide, weight_cols = prep_weights_wide(weights_raw)

    # ------------------------------------------------------------
    # 3) Gold daily features
    # ------------------------------------------------------------
    gold = live_storage.storage.read_parquet(gold_path)
    gold_wide = prep_gold_wide(
        gold,
        drop_cols={"open", "high", "low", "close", "volume"},
    )
    gold_wide = prep_state_ts(gold_wide)  # ensure session index if not already


    # ------------------------------------------------------------
    # 4) Model meta snapshots
    # ------------------------------------------------------------
    meta_raw = live_storage.read_all_model_meta(strategy=strategy, universe=universe)
    meta_sd = prep_meta_ts(meta_raw)
    # ------------------------------------------------------------
    # 5) Merge on session_date
    # ------------------------------------------------------------
    merged = portfolio
    merged = join_daily(merged, weights_wide)
    merged = join_daily(merged, meta_sd)
    merged = join_daily(merged, gold_wide)

    # ------------------------------------------------------------
    # 6) Forward-fill in-force columns (generic)
    # ------------------------------------------------------------
    merged = fill_in_force(merged)

    # ------------------------------------------------------------
    # 7) Returns + metrics
    # ------------------------------------------------------------
    merged["strategy_ret"] = _equity_to_returns(merged, equity_col="portfolio_value")

    ann_factor = int(execution_cfg.get("perf", {}).get("ann_factor", 252))
    metrics = compute_metrics_simple(merged["strategy_ret"], ann_factor=ann_factor)

    # ------------------------------------------------------------
    # 8) Persist
    # ------------------------------------------------------------
    live_storage.write_portfolio_performance(
        strategy=strategy,
        universe=universe,
        df=merged,
        metrics=metrics,
    )

    return {"metrics": metrics, "merged": merged}
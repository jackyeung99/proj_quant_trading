from __future__ import annotations

from typing import Any, Dict
import pandas as pd

from qbt.core.logging import get_logger
from qbt.execution.alpaca_client import AlpacaTradingAPI
from qbt.storage.artifacts import LiveStore
from qbt.metrics.summary import compute_metrics_simple
from qbt.utils.dates import _ensure_session_index, _merge_asof_left

logger = get_logger(__name__)





def _equity_to_returns(df: pd.DataFrame, *, equity_col: str = "portfolio_value") -> pd.Series:
    eq = pd.to_numeric(df[equity_col], errors="coerce").astype(float)
    return eq.pct_change().fillna(0.0)


def _prep_weights_wide(weights: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Keep only session_date + asset weights.
    Rename asset columns to '<asset>_weight'.
    """
    w = _ensure_session_index(weights)
    w = w.sort_index()

    meta_cols = {"generated_at_utc", "asof_utc", "config_hash", "market_tz", "cutoff_hour"}
    asset_cols = [c for c in w.columns if c not in meta_cols]

    # numeric + fill
    for c in asset_cols:
        w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0)

    # rename to <asset>_weight
    rename_map = {c: f"{c}_weight" for c in asset_cols}
    w = w.rename(columns=rename_map)
    renamed_cols = list(rename_map.values())

    # keep only weight cols, drop duplicate index entries (keep last snapshot for that session)
    w = w[renamed_cols]
    w = w[~w.index.duplicated(keep="last")]

    # (optional) ensure float dtype
    w = w.astype(float)

    return w, renamed_cols

def _session_index_naive_midnight(idx: pd.Index) -> pd.DatetimeIndex:
    """
    Canonical session index:
      - DatetimeIndex
      - midnight
      - tz-naive
    """
    di = pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce"))
    if di.tz is not None:
        di = di.tz_convert("UTC").tz_localize(None)   # drop tz safely
    return di.normalize()

def _prep_meta(meta_ts: pd.DataFrame) -> pd.DataFrame:
    if meta_ts is None or meta_ts.empty:
        return pd.DataFrame()

    return meta_ts


def _prep_gold(
    gold: pd.DataFrame,
    *,
    session_col: str = "session_date",
    ticker_col: str = "ticker",
    drop_cols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Convert long gold table into wide format:
        index = session_date
        columns = <ticker>_<feature>

    Assumes gold has columns:
        session_date, ticker, feature1, feature2, ...

    Returns:
        Wide DataFrame indexed by session_date.
    """

    if gold is None or gold.empty:
        return pd.DataFrame()

    df = gold.copy()

    # ---- normalize session_date ----
    df[session_col] = pd.to_datetime(df[session_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[session_col, ticker_col])

    # ---- choose value columns ----
    id_cols = {session_col, ticker_col}
    if drop_cols:
        id_cols |= set(drop_cols)

    value_cols = [c for c in df.columns if c not in id_cols]

    # ---- pivot ----
    wide = (
        df.pivot_table(
            index=session_col,
            columns=ticker_col,
            values=value_cols,
            aggfunc="last",
        )
        .sort_index()
    )

    # ---- flatten columns ----
    if isinstance(wide.columns, pd.MultiIndex):
        # (feature, ticker) → ticker_feature
        wide.columns = [f"{t}_{f}" for f, t in wide.columns]

    wide.index.name = session_col

    return wide

def _prep_equity(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # normalize naming
    if "equity" in x.columns and "portfolio_value" not in x.columns:
        x = x.rename(columns={"equity": "portfolio_value"})

    x = (
        x.rename(columns={"timestamp": "session_date"})
         .set_index("session_date")
         .sort_index()
    )

    x["portfolio_value"] = pd.to_numeric(x["portfolio_value"], errors="coerce")

    # -------------------------------------------------
    # Drop rows before first non-zero equity
    # -------------------------------------------------
    mask_started = x["portfolio_value"] != 0.0

    if mask_started.any():
        first_valid = mask_started.idxmax()  # first True
        x = x.loc[first_valid:]
    else:
        return x.iloc[0:0]  # empty if never funded

    return x[["portfolio_value"]]

def evaluate_portfolio(storage: LiveStore, execution_cfg: dict) -> dict:
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

    portfolio = client.get_historical_equity(
        period=period,
        timeframe=timeframe,
        extended_hours=False,
        as_df=True,
        tz="UTC",
        date_end=end_iso,
    )

    portfolio = _prep_equity(portfolio)

    # ------------------------------------------------------------
    # 2) Weights snapshots
    # ------------------------------------------------------------
    weights_raw = storage.read_all_weights(strategy=strategy, universe=universe)
    weights_wide, asset_cols = _prep_weights_wide(weights_raw)

    # ------------------------------------------------------------
    # 3) Gold daily features
    # ------------------------------------------------------------
    gold = storage.storage.read_parquet(gold_path)

    gold_wide = _prep_gold(
        gold,
        drop_cols={"open", "high", "low", "close", "volume"},
    )

    # ------------------------------------------------------------
    # 4) Model meta snapshots
    # ------------------------------------------------------------
    meta_raw = storage.read_all_model_meta(strategy=strategy, universe=universe)
    meta_sd = _prep_meta(meta_raw)

    # ------------------------------------------------------------
    # 5) Merge (asof on session_date index)
    # ------------------------------------------------------------
    merged = portfolio

    if not weights_wide.empty:
        merged = _merge_asof_left(merged, weights_wide)

    if not meta_raw.empty:
        merged = _merge_asof_left(merged, meta_raw)

    if not gold_wide.empty:
        merged = _merge_asof_left(merged, gold_wide)

    # ------------------------------------------------------------
    # 6) Forward-fill in-force quantities
    # ------------------------------------------------------------
    for col in asset_cols:
        if col in merged.columns:
            merged[col] = (
                pd.to_numeric(merged[col], errors="coerce")
                .ffill()
                .fillna(0.0)
            )

    meta_cols = [
        "tau_star",
        "w_low",
        "w_high",
        "state_var",
        "weight_allocation",
        "config_hash",
    ]

    for col in meta_cols:
        if col in merged.columns:
            if col == "config_hash":
                merged[col] = merged[col].astype("string").ffill()
            else:
                merged[col] = (
                    pd.to_numeric(merged[col], errors="coerce")
                    .ffill()
                )

    # ------------------------------------------------------------
    # 7) Returns + metrics
    # ------------------------------------------------------------
    merged["ret"] = _equity_to_returns(
        merged,
        equity_col="portfolio_value",
    )

    ann_factor = int(execution_cfg.get("perf", {}).get("ann_factor", 252))
    metrics = compute_metrics_simple(
        merged["ret"],
        ann_factor=ann_factor,
    )

    # ------------------------------------------------------------
    # 8) Persist
    # ------------------------------------------------------------

    storage.write_portfolio_performance(
        strategy=strategy,
        universe=universe,
        df=merged,
        metrics=metrics
    )


    return {
        "metrics": metrics,
        "merged": merged,
    }
from __future__ import annotations

from typing import Any, Dict, Iterable
import pandas as pd

from qbt.core.logging import get_logger
from qbt.execution.alpaca_client import AlpacaTradingAPI
from qbt.storage.artifacts import LiveStore
from qbt.metrics.summary import compute_portfolio_metrics
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
# Logging helpers
# ---------------------------------------------------------------------

def _df_brief(df: pd.DataFrame | None, name: str) -> str:
    if df is None:
        return f"{name}=None"
    if df.empty:
        return f"{name}=empty shape={df.shape}"
    idx = df.index
    idx_name = getattr(idx, "name", None)
    try:
        imin = idx.min()
        imax = idx.max()
        irng = f"{imin}..{imax}"
    except Exception:
        irng = "n/a"
    return f"{name} shape={df.shape} idx={idx_name} range={irng} cols={len(df.columns)}"


def _log_df_head(df: pd.DataFrame, name: str, n: int = 3) -> None:
    if df is None or df.empty:
        return
    try:
        logger.debug("%s head(%d):\n%s", name, n, df.head(n).to_string())
    except Exception:
        # avoid logging explosions for weird objects
        logger.debug("%s head(%d): <unprintable>", name, n)


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
        logger.info("prep_state_ts: input empty -> returning empty df")
        return pd.DataFrame()

    x = _ensure_session_index(df).sort_index()

    if rename:
        before_cols = set(x.columns)
        x = x.rename(columns=rename)
        after_cols = set(x.columns)
        logger.debug("prep_state_ts: renamed %d cols (before=%d after=%d)", len(rename), len(before_cols), len(after_cols))

    if keep_cols is not None:
        cols = [c for c in keep_cols if c in x.columns]
        missing = [c for c in keep_cols if c not in x.columns]
        if missing:
            logger.warning("prep_state_ts: missing keep_cols=%s", missing)
        x = x[cols]
    elif drop_cols:
        drop_cols = set(drop_cols)
        dropped = [c for c in x.columns if c in drop_cols]
        x = x[[c for c in x.columns if c not in drop_cols]]
        if dropped:
            logger.debug("prep_state_ts: dropped cols=%s", dropped)

    # keep last snapshot for that session_date
    dup = x.index.duplicated(keep="last")
    if dup.any():
        logger.info("prep_state_ts: dropping %d duplicated index rows (keep last)", int(dup.sum()))
    x = x[~dup]

    logger.debug(_df_brief(x, "prep_state_ts(out)"))
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
        logger.warning("prep_weights_wide: no weights data")
        return pd.DataFrame(), []

    asset_cols = [c for c in w.columns if c not in stamp_cols]
    if not asset_cols:
        logger.warning("prep_weights_wide: found no asset columns after stamp filtering; cols=%s", list(w.columns))
        return pd.DataFrame(), []

    # numeric + fill
    bad = 0
    for c in asset_cols:
        before_na = int(pd.isna(w[c]).sum())
        w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0)
        after_na = int(pd.isna(w[c]).sum())
        if after_na > 0:
            bad += 1
        if before_na > 0:
            logger.debug("prep_weights_wide: %s had %d NA -> filled", c, before_na)

    rename_map = {c: f"{c}_weight" for c in asset_cols}
    w = w.rename(columns=rename_map)

    weight_cols = list(rename_map.values())
    w = w[weight_cols].astype(float)

    logger.info(
        "prep_weights_wide: %d assets -> %d weight cols; %s..%s",
        len(asset_cols),
        len(weight_cols),
        w.index.min() if not w.empty else None,
        w.index.max() if not w.empty else None,
    )
    logger.debug(_df_brief(w, "weights_wide"))
    return w, weight_cols


def prep_meta_ts(meta: pd.DataFrame | None) -> pd.DataFrame:
    """
    Meta is already "meta columns"; we only ensure session index + dedupe.
    You may drop large nested objects here if present.
    """
    drop = set()
    if meta is not None and not meta.empty and "strategy_meta" in meta.columns:
        drop.add("strategy_meta")
        logger.info("prep_meta_ts: dropping large column strategy_meta")

    out = prep_state_ts(meta, drop_cols=drop if drop else None)
    logger.debug(_df_brief(out, "meta_ts"))
    return out


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
        logger.warning("prep_gold_wide: gold empty")
        return pd.DataFrame()

    df = gold.copy()

    df[session_col] = pd.to_datetime(df[session_col], errors="coerce").dt.normalize()
    before = len(df)
    df = df.dropna(subset=[session_col, ticker_col])
    dropped = before - len(df)
    if dropped:
        logger.info("prep_gold_wide: dropped %d rows missing session/ticker", dropped)

    id_cols = {session_col, ticker_col}
    if drop_cols:
        id_cols |= set(drop_cols)

    value_cols = [c for c in df.columns if c not in id_cols]
    if not value_cols:
        logger.warning("prep_gold_wide: no value cols after dropping ids; cols=%s", list(df.columns))
        return pd.DataFrame()

    logger.info(
        "prep_gold_wide: pivoting long->wide rows=%d tickers~=%d values=%d",
        len(df),
        df[ticker_col].nunique(dropna=True),
        len(value_cols),
    )

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
    logger.debug(_df_brief(wide, "gold_wide"))
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

    x = (
        x.rename(columns={"timestamp": "session_date"})
         .set_index("session_date")
         .sort_index()
    )

    x["portfolio_value"] = pd.to_numeric(x["portfolio_value"], errors="coerce")

    mask_started = x["portfolio_value"] != 0.0
    if mask_started.any():
        first_valid = mask_started.idxmax()
        if first_valid != x.index.min():
            logger.info("prep_equity_panel: trimming warmup zeros up to %s", first_valid)
        x = x.loc[first_valid:]
    else:
        logger.warning("prep_equity_panel: portfolio_value all zeros -> returning empty")
        return x.iloc[0:0]

    logger.debug(_df_brief(x[["portfolio_value"]], "portfolio_panel"))
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
    strategy = execution_cfg.get("strategy_name")
    universe = execution_cfg.get("universe")
    gold_path = execution_cfg.get("gold_path")

    logger.info("evaluate_portfolio: start strategy=%s universe=%s gold_path=%s", strategy, universe, gold_path)

    if not strategy or not universe:
        logger.warning("evaluate_portfolio: missing strategy_name or universe in execution_cfg")

    client = AlpacaTradingAPI(cfg=execution_cfg.get("alpaca", {}) or {})
    logger.debug("evaluate_portfolio: Alpaca client initialized")

    # ------------------------------------------------------------
    # 1) Portfolio history (Alpaca)
    # ------------------------------------------------------------
    end_utc = pd.Timestamp.now(tz="UTC")
    end_iso = end_utc.isoformat()

    port_cfg = execution_cfg.get("portfolio", {}) or {}
    period = port_cfg.get("period", "2M")
    timeframe = port_cfg.get("timeframe", "1D")

    logger.info(
        "evaluate_portfolio: fetching Alpaca equity period=%s timeframe=%s date_end=%s",
        period, timeframe, end_iso
    )

    portfolio_raw = client.get_historical_equity(
        period=period,
        timeframe=timeframe,
        extended_hours=False,
        as_df=True,
        tz="UTC",
        date_end=end_iso,
    )

    logger.debug(_df_brief(portfolio_raw, "portfolio_raw"))
    _log_df_head(portfolio_raw, "portfolio_raw")

    market_tz = execution_cfg.get("market_tz", "America/New_York")
    try:
        ts_utc = pd.to_datetime(portfolio_raw["timestamp"], utc=True, errors="coerce")
        portfolio_raw = portfolio_raw.copy()
        portfolio_raw["timestamp"] = ts_utc.dt.tz_convert(market_tz).dt.normalize().dt.tz_localize(None)
        # logger.info("evaluate_portfolio: converted timestamps UTC -> %s (normalized)", market_tz)
    except Exception as e:
        logger.exception("evaluate_portfolio: failed timestamp conversion to market session labels: %s", e)

    portfolio = prep_equity_panel(portfolio_raw)
    logger.info(_df_brief(portfolio, "portfolio"))

    # ------------------------------------------------------------
    # 2) Weights snapshots
    # ------------------------------------------------------------
    logger.info("evaluate_portfolio: reading weights snapshots")
    weights_raw = live_storage.read_all_weights(strategy=strategy, universe=universe)
    logger.debug(_df_brief(weights_raw, "weights_raw"))

    weights_wide, weight_cols = prep_weights_wide(weights_raw)
    logger.info("evaluate_portfolio: weights_wide cols=%d", len(weight_cols))

    # ------------------------------------------------------------
    # 3) Gold daily features
    # ------------------------------------------------------------
    logger.info("evaluate_portfolio: reading gold parquet %s", gold_path)
    gold = live_storage.storage.read_parquet(gold_path)
    logger.debug(_df_brief(gold, "gold_long"))

    # gold_wide = prep_gold_wide(
    #     gold,
    #     drop_cols={"open", "high", "low", "close", "volume"},
    # )

    gold_wide = gold
    gold_wide = prep_state_ts(gold_wide)  # ensure session index if not already
    logger.info(_df_brief(gold_wide, "gold_wide"))

    # ------------------------------------------------------------
    # 4) Model meta snapshots
    # ------------------------------------------------------------
    logger.info("evaluate_portfolio: reading model meta snapshots")
    meta_raw = live_storage.read_all_model_meta(strategy=strategy, universe=universe)
    logger.debug(_df_brief(meta_raw, "meta_raw"))

    meta_sd = prep_meta_ts(meta_raw)
    logger.info(_df_brief(meta_sd, "meta_sd"))

    # ------------------------------------------------------------
    # 5) Merge on session_date
    # ------------------------------------------------------------
    merged = portfolio

    logger.info("evaluate_portfolio: merging portfolio + weights")
    merged = join_daily(merged, weights_wide)
    logger.debug(_df_brief(merged, "merged(after weights)"))

    logger.info("evaluate_portfolio: merging + meta")
    merged = join_daily(merged, meta_sd)
    logger.debug(_df_brief(merged, "merged(after meta)"))

    logger.info("evaluate_portfolio: merging + gold")
    merged = join_daily(merged, gold_wide)
    logger.debug(_df_brief(merged, "merged(after gold)"))

    # quick diagnostics: missing weights coverage
    if weight_cols:
        miss_any = merged[weight_cols].isna().all(axis=1) if set(weight_cols).issubset(merged.columns) else None
        if miss_any is not None:
            n_miss = int(miss_any.sum())
            if n_miss:
                logger.warning("evaluate_portfolio: %d sessions have all-NA weights before fill_in_force", n_miss)

    # ------------------------------------------------------------
    # 6) Forward-fill in-force columns (generic)
    # ------------------------------------------------------------
    logger.info("evaluate_portfolio: fill_in_force")
    merged = fill_in_force(merged)
    logger.debug(_df_brief(merged, "merged(after fill_in_force)"))

    # ------------------------------------------------------------
    # 7) Returns + metrics
    # ------------------------------------------------------------
    logger.info("evaluate_portfolio: computing returns + metrics")
    merged["strategy_ret"] = _equity_to_returns(merged, equity_col="portfolio_value")

    
    ann_factor = int(execution_cfg.get("perf", {}).get("ann_factor", 252))
    metrics = compute_portfolio_metrics(
            merged,
            col_gross='strategy_ret',
            ann_factor=252,
            return_type="simple",  # or "log" if needed
            col_signal="signal",   # your 0/1 column
            col_xle_weight='XLE_weight'
        )

    logger.info("evaluate_portfolio: metrics computed (ann_factor=%d)", ann_factor)
    try:
        logger.debug("evaluate_portfolio: metrics=%s", metrics)
    except Exception:
        logger.debug("evaluate_portfolio: metrics=<unprintable>")

    # ------------------------------------------------------------
    # 8) Persist
    # ------------------------------------------------------------
    logger.info("evaluate_portfolio: writing portfolio performance artifacts")
    live_storage.write_portfolio_performance(
        strategy=strategy,
        universe=universe,
        df=merged,
        metrics=metrics,
    )
    logger.info("evaluate_portfolio: done strategy=%s universe=%s rows=%d cols=%d", strategy, universe, len(merged), len(merged.columns))

    return {"metrics": metrics, "merged": merged}
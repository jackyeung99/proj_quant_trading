from __future__ import annotations

from typing import Any, Dict
import pandas as pd

from qbt.core.logging import get_logger
from qbt.config.specs import StrategySpec
from qbt.storage.artifacts import LiveStore
from qbt.metrics.summary import compute_portfolio_metrics
from qbt.data.merge import join_daily, fill_in_force
from qbt.utils.dates import _ensure_session_index


logger = get_logger(__name__)


DEFAULT_STAMP_COLS: set[str] = {
    "session_date",
    "asof_utc",
    "generated_at_utc",
    "trained_at_utc",
    "trained_at",
    "train_start_ts_utc",
    "train_end_ts_utc",
    "config_hash",
    "snapshot_id",
    "model_snapshot_id",
    "market_tz",
    "cutoff_hour",
    "period",
    "timeframe",
    "extended_hours",
    "base_value",
}


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
        logger.debug("%s head(%d): <unprintable>", name, n)


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
    if df is None or df.empty:
        logger.info("prep_state_ts: input empty -> returning empty df")
        return pd.DataFrame()

    x = _ensure_session_index(df).sort_index()

    if rename:
        before_cols = set(x.columns)
        x = x.rename(columns=rename)
        after_cols = set(x.columns)
        logger.debug(
            "prep_state_ts: renamed %d cols (before=%d after=%d)",
            len(rename), len(before_cols), len(after_cols)
        )

    if keep_cols is not None:
        cols = [c for c in keep_cols if c in x.columns]
        missing = [c for c in keep_cols if c not in x.columns]
        if missing:
            logger.warning("prep_state_ts: missing keep_cols=%s", missing)
        x = x[cols]
    elif drop_cols:
        dropped = [c for c in x.columns if c in drop_cols]
        x = x[[c for c in x.columns if c not in drop_cols]]
        if dropped:
            logger.debug("prep_state_ts: dropped cols=%s", dropped)

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
    w = prep_state_ts(weights)
    if w.empty:
        logger.warning("prep_weights_wide: no weights data")
        return pd.DataFrame(), []

    asset_cols = [c for c in w.columns if c not in stamp_cols]
    if not asset_cols:
        logger.warning("prep_weights_wide: found no asset columns after stamp filtering; cols=%s", list(w.columns))
        return pd.DataFrame(), []

    for c in asset_cols:
        before_na = int(pd.isna(w[c]).sum())
        w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0)
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
    drop = set()
    if meta is not None and not meta.empty and "strategy_meta" in meta.columns:
        drop.add("strategy_meta")
        logger.info("prep_meta_ts: dropping large column strategy_meta")

    out = prep_state_ts(meta, drop_cols=drop if drop else None)
    logger.debug(_df_brief(out, "meta_ts"))
    return out


def prep_equity_panel(df: pd.DataFrame) -> pd.DataFrame:
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


def _resolve_input_table_path(
    live_storage: LiveStore,
    strategy: StrategySpec,
) -> str:
    if not strategy.input_table:
        raise ValueError("strategy.input_table must be set.")

    if not hasattr(live_storage, "paths") or live_storage.paths is None:
        raise ValueError("live_storage.paths is required to resolve strategy.input_table")

    return live_storage.paths.gold_table_key(
        freq=strategy.input_table_freq,
        tag=strategy.input_table,
    )


def evaluate_portfolio(
    client: Any,
    live_storage: LiveStore,
    strategy: StrategySpec,
    *,
    gold_path: str | None = None,
) -> dict[str, Any]:
    """
    Evaluation step:
      1) fetch portfolio equity history
      2) load weights snapshots, model meta snapshots, gold features
      3) canonicalize & merge on session_date
      4) forward-fill in-force columns
      5) compute returns & metrics
      6) persist panel + metrics
    """
    strategy_name = strategy.strategy_name
    universe = strategy.universe

    evaluation_cfg = strategy.evaluation or {}
    execution_cfg = strategy.execution or {}

    market_tz = execution_cfg.get("market_tz", "America/New_York")
    resolved_gold_path = _resolve_input_table_path(live_storage, strategy)

    logger.info(
        "evaluate_portfolio: start strategy=%s universe=%s gold_path=%s",
        strategy_name,
        universe,
        resolved_gold_path,
    )

    portfolio_cfg = evaluation_cfg.get("portfolio", {}) or {}
    period = portfolio_cfg.get("period", "2M")
    timeframe = portfolio_cfg.get("timeframe", "1D")

    end_utc = pd.Timestamp.now(tz="UTC")
    end_iso = end_utc.isoformat()

    logger.info(
        "evaluate_portfolio: fetching Alpaca equity period=%s timeframe=%s date_end=%s",
        period,
        timeframe,
        end_iso,
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

    try:
        ts_utc = pd.to_datetime(portfolio_raw["timestamp"], utc=True, errors="coerce")
        portfolio_raw = portfolio_raw.copy()
        portfolio_raw["timestamp"] = (
            ts_utc.dt.tz_convert(market_tz)
            .dt.normalize()
            .dt.tz_localize(None)
        )
    except Exception as e:
        logger.exception(
            "evaluate_portfolio: failed timestamp conversion to market session labels: %s",
            e,
        )

    portfolio = prep_equity_panel(portfolio_raw)
    logger.info(_df_brief(portfolio, "portfolio"))

    logger.info("evaluate_portfolio: reading weights snapshots")
    weights_raw = live_storage.read_all_weights(strategy=strategy_name, universe=universe)
    logger.debug(_df_brief(weights_raw, "weights_raw"))

    weights_wide, weight_cols = prep_weights_wide(weights_raw)
    logger.info("evaluate_portfolio: weights_wide cols=%d", len(weight_cols))

    logger.info("evaluate_portfolio: reading gold parquet %s", resolved_gold_path)
    gold = live_storage.storage.read_parquet(resolved_gold_path)
    logger.debug(_df_brief(gold, "gold_raw"))

    gold_wide = prep_state_ts(gold)
    logger.info(_df_brief(gold_wide, "gold_wide"))

    logger.info("evaluate_portfolio: reading model meta snapshots")
    meta_raw = live_storage.read_all_model_meta(strategy=strategy_name, universe=universe)
    logger.debug(_df_brief(meta_raw, "meta_raw"))

    meta_sd = prep_meta_ts(meta_raw)
    logger.info(_df_brief(meta_sd, "meta_sd"))

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

    if weight_cols and set(weight_cols).issubset(merged.columns):
        miss_any = merged[weight_cols].isna().all(axis=1)
        n_miss = int(miss_any.sum())
        if n_miss:
            logger.warning(
                "evaluate_portfolio: %d sessions have all-NA weights before fill_in_force",
                n_miss,
            )

    logger.info("evaluate_portfolio: fill_in_force")
    merged = fill_in_force(merged)
    logger.debug(_df_brief(merged, "merged(after fill_in_force)"))

    logger.info("evaluate_portfolio: computing returns + metrics")
    merged["strategy_ret"] = _equity_to_returns(merged, equity_col="portfolio_value")

    perf_cfg = evaluation_cfg.get("perf", {}) or {}
    ann_factor = int(perf_cfg.get("ann_factor", 252))
    benchmark_col = perf_cfg.get("benchmark_col", "SPY_ret_cc")
    signal_col = perf_cfg.get("signal_col", "signal")

    metrics = compute_portfolio_metrics(
        merged,
        col_gross="strategy_ret",
        ann_factor=ann_factor,
        return_type="simple",
        col_signal=signal_col,
        col_bh=benchmark_col,
    )

    logger.info("evaluate_portfolio: metrics computed (ann_factor=%d)", ann_factor)
    try:
        logger.debug("evaluate_portfolio: metrics=%s", metrics)
    except Exception:
        logger.debug("evaluate_portfolio: metrics=<unprintable>")

    logger.info("evaluate_portfolio: writing portfolio performance artifacts")
    live_storage.write_portfolio_performance(
        strategy=strategy_name,
        universe=universe,
        df=merged,
        metrics=metrics,
    )

    logger.info(
        "evaluate_portfolio: done strategy=%s universe=%s rows=%d cols=%d",
        strategy_name,
        universe,
        len(merged),
        len(merged.columns),
    )

    return {"metrics": metrics, "merged": merged}
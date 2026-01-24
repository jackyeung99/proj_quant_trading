# dashboard/callbacks.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx

from dashboard.services.data_access import (
    load_runs,
    read_timeseries_for_run,
    read_meta_for_run,
    meta_params_table,
    safe_read_metrics,
    StoreContext,
)


from ..layout.components import badge_row, kv_panel

from qbt.metrics.plots import PLOT_REGISTRY


def _df_to_dash(df: pd.DataFrame):
    return df.to_dict("records"), [{"name": c, "id": c} for c in df.columns]


def _compute_kpis(ts: pd.DataFrame) -> dict:
    ts = ts.copy()
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index, errors="coerce")
    ts = ts.sort_index()

    if "equity_net" in ts.columns:
        eq = ts["equity_net"].astype(float)
        ret = eq.pct_change()
    else:
        rc = "ret_net" if "ret_net" in ts.columns else ("ret" if "ret" in ts.columns else None)
        if rc is None:
            return dict(total_return=None, cagr=None, vol=None, mdd=None, sharpe=None)
        ret = ts[rc].astype(float)
        eq = (1.0 + ret.fillna(0.0)).cumprod()

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else None

    ann = 252.0
    r = ret.dropna()
    vol = float(r.std(ddof=0) * np.sqrt(ann)) if r.shape[0] > 2 else None
    sharpe = float((r.mean() / r.std(ddof=0)) * np.sqrt(ann)) if r.shape[0] > 2 and r.std(ddof=0) > 0 else None

    if len(eq) > 1 and pd.notna(ts.index[0]) and pd.notna(ts.index[-1]):
        years = (ts.index[-1] - ts.index[0]).days / 365.25
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years and years > 0 else None
    else:
        cagr = None

    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min()) if len(dd) else None

    return dict(total_return=total_return, cagr=cagr, vol=vol, mdd=mdd, sharpe=sharpe)


def register_callbacks(app, store_ctx: StoreContext):

    # ------------------------------------------------------------
    # Populate strategy dropdown + run dropdown + runs table
    # ------------------------------------------------------------
    @app.callback(
        Output("strategy_dd", "options"),
        Output("strategy_dd", "value"),
        Output("run_dd", "options"),
        Output("run_dd", "value"),
        Output("runs_table", "data"),
        Output("runs_table", "columns"),
        Output("status", "children"),
        Output("status", "color"),
        Input("strategy_dd", "id"),
        Input("strategy_dd", "value"),
        Input("runs_table", "active_cell"),
        State("runs_table", "data"),
    )
    def init_and_filter(_, strategy_value, active_cell, table_rows):
        runs = load_runs(store_ctx)
        if runs.empty:
            return [], None, [], None, [], [], "No runs found. Run: python scripts/run_backtest.py", "warning"

        strategies = sorted(runs["strategy_name"].dropna().unique()) if "strategy_name" in runs.columns else []
        strat_opts = [{"label": s, "value": s} for s in strategies]
        strat_val = strategy_value if strategy_value in strategies else None

        runs_f = runs.copy()
        if strat_val:
            runs_f = runs_f.loc[runs_f["strategy_name"] == strat_val].copy()

        if runs_f.empty:
            return strat_opts, strat_val, [], None, [], [], "No runs after filter.", "warning"

        if "label" not in runs_f.columns:
            cols = [c for c in ["run_id", "strategy_name", "universe"] if c in runs_f.columns]
            runs_f["label"] = runs_f[cols].astype(str).agg(" | ".join, axis=1)

        run_opts = [{"label": r["label"], "value": r["run_id"]} for _, r in runs_f.iterrows()]
        default_run_id = run_opts[0]["value"]

        show_cols = [c for c in ["created_at_utc", "run_id", "strategy_name", "universe"] if c in runs_f.columns]
        tv = runs_f[show_cols].copy()
        if "created_at_utc" in tv.columns:
            tv["created_at_utc"] = pd.to_datetime(tv["created_at_utc"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

        table_data = tv.to_dict("records")
        table_cols = [{"name": c, "id": c} for c in tv.columns]

        run_val = default_run_id
        if ctx.triggered_id == "runs_table" and active_cell and table_rows:
            try:
                clicked_row = active_cell["row"]
                candidate = table_rows[clicked_row].get("run_id")
                run_val = candidate or default_run_id
            except Exception:
                run_val = default_run_id

        return strat_opts, strat_val, run_opts, run_val, table_data, table_cols, f"Loaded {len(run_opts)} run(s).", "secondary"

    # ------------------------------------------------------------
    # Selected run: badges + params + KPI cards + equity fig
    # ------------------------------------------------------------
    @app.callback(
    Output("run_badges", "children"),
    Output("params_kv", "children"),
    Output("kpi_sharpe", "children"),
    Output("kpi_cagr", "children"),
    Output("kpi_vol", "children"),
    Output("kpi_mdd", "children"),
    Output("equity_fig", "figure"),
    Input("run_dd", "value"),
)
    def update_run_details(run_id: str):
        import plotly.express as px

        # ---------- empty state ----------
        if not run_id:
            fig = px.line(title="No run selected")
            return "", "", "-", "-", "-", "-", fig

        # ---------- load timeseries ----------
        try:
            _row, ts = read_timeseries_for_run(store_ctx, run_id)
        except Exception as e:
            fig = px.line(title="Unable to load timeseries")
            badges = badge_row([("Run", run_id), ("Error", "timeseries load failed")])
            params_component = kv_panel([("timeseries_error", repr(e))])
            return badges, params_component, "-", "-", "-", "-", fig

        if not isinstance(ts.index, pd.DatetimeIndex):
            ts = ts.copy()
            ts.index = pd.to_datetime(ts.index, errors="coerce")
        ts = ts.sort_index()

        # ---------- load meta ----------
        meta_err = None
        try:
            meta = read_meta_for_run(store_ctx, run_id)
            if not isinstance(meta, dict):
                raise TypeError(f"meta must be dict, got {type(meta)}")
        except Exception as e:
            meta = {}
            meta_err = repr(e)

        # ---------- badges ----------
        strategy = meta.get("strategy_name") or (meta.get("spec") or {}).get("strategy_name") or ""
        universe = meta.get("universe") or (meta.get("spec") or {}).get("universe") or ""
        created = meta.get("created_at_utc") or meta.get("created_at") or ""
        badges_list = [("Run", run_id), ("Strategy", strategy), ("Universe", universe), ("Created", created)]
        if meta_err:
            badges_list.append(("Meta", "FAILED"))
        badges_comp = badge_row(badges_list)

        # ---------- params kv ----------
        # meta_params_table returns a df with columns ["parameter","value"] (ideally)
        try:
            params_df = meta_params_table(meta)
        except Exception as e:
            params_df = pd.DataFrame([{"parameter": "params_error", "value": repr(e)}])

        # Make meta errors visible
        if meta_err:
            params_df = pd.concat(
                [pd.DataFrame([{"parameter": "meta_error", "value": meta_err}]), params_df],
                ignore_index=True,
            )

        # Normalize to parameter/value even if df is weird
        if params_df.empty:
            items = [("info", "No parameters found")]
        else:
            if not {"parameter", "value"}.issubset(params_df.columns):
                # fallback: use first two columns
                cols = list(params_df.columns)
                if len(cols) >= 2:
                    params_df = params_df.rename(columns={cols[0]: "parameter", cols[1]: "value"})
                elif len(cols) == 1:
                    params_df = params_df.rename(columns={cols[0]: "value"})
                    params_df["parameter"] = "value"
                else:
                    params_df = pd.DataFrame([{"parameter": "info", "value": "No parameters found"}])

            items = [(str(r["parameter"]), str(r["value"])) for r in params_df[["parameter", "value"]].to_dict("records")]

        params_component = kv_panel(items)

        # ---------- KPIs ----------
        k = _compute_kpis(ts)

        def fmt_pct(x):
            return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.2f}%"

        def fmt_num(x):
            return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"

        kpi_sharpe = fmt_num(k.get("sharpe"))
        kpi_cagr = fmt_pct(k.get("cagr"))
        kpi_vol = fmt_pct(k.get("vol"))
        kpi_mdd = fmt_pct(k.get("mdd"))

        # ---------- equity plot ----------
        plot_fn = PLOT_REGISTRY.get("equity")
        if plot_fn is not None:
            fig = plot_fn(ts=ts, run_id=run_id, meta=meta)
        else:
            if "equity_net" in ts.columns:
                fig = px.line(ts, y="equity_net", title=f"Equity: {run_id}")
            else:
                fig = px.line(title="Equity unavailable")

        return badges_comp, params_component, kpi_sharpe, kpi_cagr, kpi_vol, kpi_mdd, fig

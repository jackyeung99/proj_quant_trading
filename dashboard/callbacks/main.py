import pandas as pd
import plotly.express as px
from dash import Input, Output, State, ctx

from dashboard.services.data_access import load_runs, safe_read_metrics, read_timeseries_for_run, StoreContext


def register_callbacks(app, store_ctx: StoreContext):

    @app.callback(
        Output("run_dd", "options"),
        Output("run_dd", "value"),
        Output("status", "children"),
        Output("status", "color"),
        Output("runs_table", "data"),
        Output("runs_table", "columns"),
        Input("run_dd", "id"),
        Input("runs_table", "active_cell"),
        State("runs_table", "data"),
    )
    def init_or_table_pick(_, active_cell, table_rows):
        runs = load_runs(store_ctx)
        if runs.empty:
            return [], None, "No runs found. Run: python scripts/run_backtest.py", "warning", [], []

        options = [{"label": r["label"], "value": r["run_id"]} for _, r in runs.iterrows()]
        default_run_id = options[0]["value"]

        show_cols = [c for c in ["created_at_utc", "run_id", "strategy_name", "universe"] if c in runs.columns]
        runs_view = runs[show_cols].copy()

        if "created_at_utc" in runs_view.columns:
            runs_view["created_at_utc"] = pd.to_datetime(
                runs_view["created_at_utc"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M")

        data = runs_view.to_dict("records")
        columns = [{"name": c, "id": c} for c in runs_view.columns]

        run_value = default_run_id
        if ctx.triggered_id == "runs_table" and active_cell and table_rows:
            try:
                clicked_row = active_cell["row"]
                candidate = table_rows[clicked_row].get("run_id")
                run_value = candidate or default_run_id
            except Exception:
                run_value = default_run_id

        return options, run_value, f"Loaded {len(options)} run(s).", "secondary", data, columns


    @app.callback(
        Output("equity_fig", "figure"),
        Output("metrics_table", "data"),
        Output("metrics_table", "columns"),
        Input("run_dd", "value"),
    )
    def plot_equity_and_metrics(run_id):
        if not run_id:
            fig = px.line(title="No run selected")
            return fig, [], []

        try:
            _row, ts = read_timeseries_for_run(store_ctx, run_id)
        except Exception as e:
            fig = px.line(title="Unable to load run")
            table_df = pd.DataFrame([{"metric": "error", "value": str(e)}])
            return fig, table_df.to_dict("records"), [{"name": c, "id": c} for c in table_df.columns]

        if not isinstance(ts.index, pd.DatetimeIndex):
            ts.index = pd.to_datetime(ts.index)

        fig = px.line(ts, y="equity_net", title=f"Equity (Net): {run_id}")
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), title_x=0.02)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True)

        metrics = safe_read_metrics(store_ctx)

        if not metrics.empty and "run_id" in metrics.columns:
            m = metrics.loc[metrics["run_id"] == run_id].copy()
            if m.empty:
                table_df = pd.DataFrame([{"metric": "info", "value": "No metrics found"}])
            else:
                if len(m) == 1:
                    table_df = m.drop(columns=["run_id"], errors="ignore")
                else:
                    if {"metric", "value"}.issubset(m.columns):
                        table_df = m[["metric", "value"]]
                    else:
                        table_df = m
        else:
            eq = ts["equity_net"].astype(float)
            daily_ret = eq.pct_change().dropna()
            table_df = pd.DataFrame(
                [
                    {"metric": "Start equity", "value": float(eq.iloc[0])},
                    {"metric": "End equity", "value": float(eq.iloc[-1])},
                    {"metric": "Total return", "value": float(eq.iloc[-1] / eq.iloc[0] - 1.0)},
                    {"metric": "Avg daily return", "value": float(daily_ret.mean()) if not daily_ret.empty else None},
                    {"metric": "Daily vol", "value": float(daily_ret.std()) if not daily_ret.empty else None},
                ]
            )

        data = table_df.to_dict("records")
        columns = [{"name": c, "id": c} for c in table_df.columns]
        return fig, data, columns

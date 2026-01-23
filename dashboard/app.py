from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from dash import dash_table
import dash_bootstrap_components as dbc

from qbt.storage.storage import LocalStorage
from qbt.storage.paths import StoragePaths
from qbt.storage.artifacts import ArtifactsStore

THEME = dbc.themes.LITERA
app = Dash(__name__, external_stylesheets=[THEME])

storage = LocalStorage(base_dir=Path("."))
paths = StoragePaths(root="results")
store = ArtifactsStore(storage, paths)


def load_runs() -> pd.DataFrame:
    runs = store.read_runs()
    if runs.empty:
        return runs
    runs = runs.copy()
    runs["label"] = runs["run_id"] + " | " + runs["strategy_name"] + " | " + runs["universe"]
    return runs.sort_values("created_at_utc", ascending=False)


def safe_read_metrics() -> pd.DataFrame:
    if hasattr(store, "read_metrics"):
        try:
            return store.read_metrics()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


BASE_TABLE_STYLE = {
    "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
    "fontSize": "14px",
}

app.layout = dbc.Container(
    fluid=False,
    style={"maxWidth": "1100px", "paddingTop": "32px", "paddingBottom": "48px"},
    children=[
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H2("Backtesting Results", style={"marginBottom": "4px"}),
                        html.Div("Runs, equity curve, and summary metrics.", style={"opacity": 0.7}),
                    ]
                )
            )
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Alert(id="status", color="secondary", className="py-2", style={"marginBottom": "12px"}),
                        dbc.Label("Run"),
                        dcc.Dropdown(id="run_dd", placeholder="Select a run..."),
                        html.Div(style={"height": "16px"}),
                        dbc.Card(
                            [
                                dbc.CardHeader("Runs (click a row to select)"),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="runs_table",
                                        page_size=8,
                                        sort_action="native",
                                        filter_action="native",
                                        # Use active_cell click instead of selected_rows to avoid selection state weirdness
                                        cell_selectable=True,
                                        row_selectable=False,
                                        style_table={"overflowX": "auto"},
                                        style_cell={**BASE_TABLE_STYLE, "padding": "8px", "whiteSpace": "nowrap"},
                                        style_header={**BASE_TABLE_STYLE, "fontWeight": "600"},
                                        style_data_conditional=[
                                            {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"}
                                        ],
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Equity Curve"),
                                dbc.CardBody(dcc.Graph(id="equity_fig", config={"displayModeBar": False})),
                            ],
                            className="shadow-sm",
                        ),
                        html.Div(style={"height": "16px"}),
                        dbc.Card(
                            [
                                dbc.CardHeader("Selected Run Metrics"),
                                dbc.CardBody(
                                    dash_table.DataTable(
                                        id="metrics_table",
                                        page_size=10,
                                        style_table={"overflowX": "auto"},
                                        style_cell={**BASE_TABLE_STYLE, "padding": "8px"},
                                        style_header={**BASE_TABLE_STYLE, "fontWeight": "600"},
                                        style_data_conditional=[
                                            {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"}
                                        ],
                                    )
                                ),
                            ],
                            className="shadow-sm",
                        ),
                    ],
                    md=7,
                ),
            ],
            className="g-3",
        ),
    ],
)


@app.callback(
    Output("run_dd", "options"),
    Output("run_dd", "value"),
    Output("status", "children"),
    Output("status", "color"),
    Output("runs_table", "data"),
    Output("runs_table", "columns"),
    Input("run_dd", "id"),                 # init trigger
    Input("runs_table", "active_cell"),    # click trigger
    State("runs_table", "data"),
)
def init_or_table_pick(_, active_cell, table_rows):
    runs = load_runs()
    if runs.empty:
        return [], None, "No runs found. Run: python scripts/run_backtest.py", "warning", [], []

    options = [{"label": r["label"], "value": r["run_id"]} for _, r in runs.iterrows()]
    default_run_id = options[0]["value"]

    # Build table
    show_cols = [c for c in ["created_at_utc", "run_id", "strategy_name", "universe"] if c in runs.columns]
    runs_view = runs[show_cols].copy()
    if "created_at_utc" in runs_view.columns:
        runs_view["created_at_utc"] = pd.to_datetime(
            runs_view["created_at_utc"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M")

    data = runs_view.to_dict("records")
    columns = [{"name": c, "id": c} for c in runs_view.columns]

    # Decide dropdown value:
    # - on init: use default
    # - on table click: pick that row's run_id
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

    runs = store.read_runs()
    row = runs.loc[runs["run_id"] == run_id].iloc[0]
    ts = store.read_timeseries(row["strategy_name"], row["universe"], run_id)

    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index)

    fig = px.line(ts, y="equity_net", title=f"Equity (Net): {run_id}")
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), title_x=0.02)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)

    metrics = safe_read_metrics()
    if not metrics.empty and "run_id" in metrics.columns:
        m = metrics.loc[metrics["run_id"] == run_id].copy()
        if m.empty:
            table_df = pd.DataFrame([{"metric": "info", "value": "No metrics found for this run_id"}])
        else:
            if len(m) == 1:
                table_df = m.drop(columns=[c for c in ["run_id"] if c in m.columns])
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


if __name__ == "__main__":
    app.run(debug=True)

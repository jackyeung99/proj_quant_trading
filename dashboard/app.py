from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

from qbt.storage.storage import LocalStorage
from qbt.storage.paths import StoragePaths
from qbt.storage.artifacts import ArtifactsStore

app = Dash(__name__)

storage = LocalStorage(base_dir=Path("."))
paths = StoragePaths(root="results")
store = ArtifactsStore(storage, paths)

def load_runs():
    runs = store.read_runs()
    if runs.empty:
        return runs
    # display label
    runs["label"] = runs["run_id"] + " | " + runs["strategy_name"] + " | " + runs["universe"]
    return runs.sort_values("created_at_utc", ascending=False)

app.layout = html.Div(
    style={"maxWidth": "900px", "margin": "40px auto", "fontFamily": "Arial"},
    children=[
        html.H2("Quant Backtest MVP: Buy & Hold"),
        html.Div(id="status"),
        dcc.Dropdown(id="run_dd", placeholder="Select a run..."),
        dcc.Graph(id="equity_fig"),
    ],
)

@app.callback(
    Output("run_dd", "options"),
    Output("run_dd", "value"),
    Output("status", "children"),
    Input("run_dd", "id"),
)
def init(_):
    runs = load_runs()
    if runs.empty:
        return [], None, "No runs found. Run: python scripts/run_backtest.py"
    options = [{"label": r["label"], "value": r["run_id"]} for _, r in runs.iterrows()]
    return options, options[0]["value"], f"Loaded {len(options)} run(s)."

@app.callback(
    Output("equity_fig", "figure"),
    Input("run_dd", "value"),
)
def plot_equity(run_id):
    if not run_id:
        return px.line(title="No run selected")

    runs = store.read_runs()
    row = runs.loc[runs["run_id"] == run_id].iloc[0]
    ts = store.read_timeseries(row["strategy_name"], row["universe"], run_id)

    # ensure index is datetime
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index)

    fig = px.line(ts, y="equity_net", title=f"Equity (Net): {run_id}")
    return fig

if __name__ == "__main__":
    app.run(debug=True)

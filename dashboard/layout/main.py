# dashboard/layout.py
import dash_bootstrap_components as dbc
from dash import dcc, html

from .components import card, kpi, spacer, nice_table

def make_layout(max_width_px: int = 1500):
    # LEFT: Filters + Runs
    controls = card(
        "Filters",
        [
            dbc.Alert(id="status", color="secondary", className="py-2", style={"marginBottom": "12px"}),

            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Label("Strategy", className="mb-1"),
                         dcc.Dropdown(id="strategy_dd", placeholder="All strategies")],
                        md=6,
                    ),
                    dbc.Col(
                        [dbc.Label("Universe", className="mb-1"),
                         dcc.Dropdown(id="universe_dd", placeholder="All universes")],
                        md=6,
                    ),
                ],
                className="g-2",
            ),
            spacer(10),

            dbc.Label("Run", className="mb-1"),
            dcc.Dropdown(id="run_dd", placeholder="Select a run..."),
        ],
    )

    runs_list = card(
        "Runs",
        [
            nice_table(
                "runs_table",
                page_size=12,
                height_px=540,
                nowrap=True,
                selectable=True,
                filter_action="none",  # cleaner
            )
        ],
    )

    left = dbc.Stack([controls, runs_list], gap=3)

    # RIGHT: Selected run (badges + kv) + KPIs + chart
    selected_run = dbc.Card(
        [
            dbc.CardHeader("Selected run", style={"fontWeight": 600, "fontSize": "0.95rem"}),
            dbc.CardBody(
                [
                    # badges row
                    html.Div(id="run_badges"),
                    spacer(8),

                    # KV label + content
                    html.Div("Model parameters", style={"fontWeight": 600, "opacity": 0.7, "fontSize": "0.85rem"}),
                    spacer(6),
                    html.Div(id="params_kv"),
                ],
                style={"padding": "14px"},
            ),
        ],
        className="shadow-sm",
    )

    kpis = dbc.Row(
        [
            dbc.Col(kpi("Sharpe", "kpi_sharpe"), md=3),
            dbc.Col(kpi("CAGR", "kpi_cagr", "Annualized"), md=3),
            dbc.Col(kpi("Max Drawdown", "kpi_mdd"), md=3),
            dbc.Col(kpi("Volatility", "kpi_vol", "Annualized"), md=3),
        ],
        className="g-3",
    )

    equity = card(
        "Equity curve",
        dcc.Graph(
            id="equity_fig",
            style={"height": "520px"},
            config={"displayModeBar": False, "displaylogo": False, "responsive": True},
        ),
    )

    right = dbc.Stack([selected_run, kpis, equity], gap=3)

    # Page
    return dbc.Container(
        fluid=True,  # allow full width
        style={
            "maxWidth": f"{max_width_px}px",
            "paddingTop": "18px",
            "paddingBottom": "38px",
        },
        children=[
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H2("Backtesting Dashboard", style={"marginBottom": "4px"}),
                            html.Div(
                                "Browse strategies and runs, inspect parameters, and compare performance.",
                                style={"opacity": 0.7},
                            ),
                        ]
                    )
                )
            ),
            html.Hr(style={"marginTop": "14px", "marginBottom": "18px"}),

            dbc.Row(
                [
                    dbc.Col(left, lg=3, md=4, sm=12),   # narrower left on laptop
                    dbc.Col(right, lg=9, md=8, sm=12),  # wide right
                ],
                className="g-4",
            ),
        ],
    )

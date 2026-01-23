import dash_bootstrap_components as dbc
from dash import dcc, html
from dash import dash_table

from .styles import BASE_TABLE_STYLE


def make_layout(max_width_px: int = 1100):
    return dbc.Container(
        fluid=False,
        style={"maxWidth": f"{max_width_px}px", "paddingTop": "32px", "paddingBottom": "48px"},
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

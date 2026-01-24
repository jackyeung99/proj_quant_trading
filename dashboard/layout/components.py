# dashboard/components.py
from __future__ import annotations
from typing import Any, Iterable, Tuple, Optional, List

import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table

from .styles import BASE_TABLE_STYLE

_ODD_ROW = [{"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"}]

def spacer(h: int = 12):
    return html.Div(style={"height": f"{h}px"})

def card(title: str, body: Any, className: str = "shadow-sm"):
    return dbc.Card(
        [
            dbc.CardHeader(title, style={"fontWeight": 600, "fontSize": "0.95rem"}),
            dbc.CardBody(body, style={"padding": "14px"}),
        ],
        className=className,
    )

def kpi(title: str, value_id: str, help_text: Optional[str] = None):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, style={"opacity": 0.7, "fontSize": "0.80rem"}),
                html.Div(id=value_id, style={"fontSize": "1.45rem", "fontWeight": 750, "lineHeight": "1.15"}),
                html.Div(help_text, style={"opacity": 0.6, "fontSize": "0.75rem"}) if help_text else None,
            ],
            style={"padding": "12px"},
        ),
        className="shadow-sm",
        style={"height": "94px"},
    )

def badge_row(items: List[Tuple[str, Any]]):
    out = []
    for label, value in items:
        if value is None or value == "":
            continue
        out.append(
            dbc.Badge(
                f"{label}: {value}",
                color="light",
                text_color="dark",
                pill=True,
                className="me-2 mb-2",
                style={"fontSize": "0.80rem", "fontWeight": 500},
            )
        )
    return html.Div(out, style={"display": "flex", "flexWrap": "wrap"})

def kv_panel(items: List[Tuple[str, str]], max_height_px: int = 320, max_rows: int = 80):
    """
    Clean key/value rendering using CSS grid.
    Prevents horizontal scroll and aligns values nicely.
    """
    rows = []
    for k, v in items[:max_rows]:
        rows.append(
            html.Div(
                [
                    html.Div(str(k), style={"fontWeight": 650, "fontSize": "0.90rem", "overflow": "hidden", "textOverflow": "ellipsis"}),
                    html.Div(str(v), style={"fontSize": "0.90rem", "opacity": 0.92, "whiteSpace": "pre-wrap", "wordBreak": "break-word"}),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "220px 1fr",  # key column fixed, value flexible
                    "gap": "12px",
                    "padding": "8px 0",
                    "borderBottom": "1px solid rgba(0,0,0,0.06)",
                },
            )
        )

    if len(items) > max_rows:
        rows.append(html.Div(f"... {len(items)-max_rows} more", style={"opacity": 0.7, "paddingTop": "6px"}))

    return html.Div(
        rows,
        style={
            "maxHeight": f"{max_height_px}px",
            "overflowY": "auto",
            "overflowX": "hidden",  
            "paddingRight": "6px",
        },
    )

def nice_table(
    table_id: str,
    page_size: int = 10,
    height_px: int = 420,
    nowrap: bool = True,
    selectable: bool = False,
    **kwargs,
):
    style_cell = {**BASE_TABLE_STYLE, "padding": "8px"}
    if nowrap:
        style_cell.update(
            {"whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis", "maxWidth": 0}
        )
    else:
        style_cell.update({"whiteSpace": "normal", "height": "auto"})

    props = dict(
        id=table_id,
        page_size=page_size,
        style_table={"overflowX": "auto", "overflowY": "auto", "height": f"{height_px}px"},
        style_cell=style_cell,
        style_header={**BASE_TABLE_STYLE, "fontWeight": "600"},
        style_data_conditional=_ODD_ROW + [{"if": {"state": "selected"}, "backgroundColor": "rgba(13, 110, 253, 0.10)"}],
        sort_action="native",
        filter_action="native",
    )

    if selectable:
        props.update(row_selectable="single", selected_rows=[])

    # let caller override safely
    props.update(kwargs)

    return dash_table.DataTable(**props)

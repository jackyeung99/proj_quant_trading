from dash import Dash

from dashboard.config import CONFIG
from dashboard.layout.main import make_layout
from dashboard.services.data_access import build_store
from dashboard.callbacks.main import register_callbacks


def create_app() -> Dash:
    app = Dash(__name__, external_stylesheets=[CONFIG.theme])
    app.layout = make_layout(CONFIG.max_width_px)

    store_ctx = build_store(CONFIG.base_dir, CONFIG.results_root)
    register_callbacks(app, store_ctx)

    return app


if __name__ == "__main__":
    create_app().run(debug=True)

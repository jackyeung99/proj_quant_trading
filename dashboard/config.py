from dataclasses import dataclass
from pathlib import Path
import dash_bootstrap_components as dbc


@dataclass(frozen=True)
class DashboardConfig:
    theme: str = dbc.themes.LITERA
    base_dir: Path = Path(".")
    results_root: str = "results"
    max_width_px: int = 1100


CONFIG = DashboardConfig()

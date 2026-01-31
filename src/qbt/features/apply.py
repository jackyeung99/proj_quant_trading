from __future__ import annotations

import pandas as pd 
import numpy as np
from typing import Any, Callable, Iterable, Mapping


def apply_transforms(df: pd.DataFrame, transforms_cfg: list[dict], registry: dict) -> pd.DataFrame:
    out = df
    for t in transforms_cfg or []:
        name = t.get("name")
        if not name:
            raise ValueError(f"Transform missing 'name': {t}")
        fn = registry.get(name)
        if fn is None:
            raise ValueError(f"Unknown transform '{name}'. Available: {sorted(registry)}")
        params = {k: v for k, v in t.items() if k != "name"}
        out = fn(out, **params)
    return out



def apply_intra_features(
    bars: pd.DataFrame,
    features: Iterable[Mapping[str, Any]],
) -> pd.Series:
    """
    Compute many intraday features on a single bucket (one day/session).
    features: iterable of mappings with keys:
      - name: str
      - requires: iterable[str]
      - func: Callable[[pd.DataFrame], float]
    Returns Series indexed by feature name.
    """
    out: dict[str, float] = {}
    cols = set(bars.columns)

    for feat in features:
        name = str(feat.get("name"))
        requires = tuple(feat.get("requires", ()))  # type: ignore
        func = feat.get("func")

        if not name or func is None:
            # bad spec -> NaN (or raise if you prefer)
            if name:
                out[name] = np.nan
            continue

        missing = [c for c in requires if c not in cols]
        if missing:
            out[name] = np.nan
            continue

        try:
            out[name] = float(func(bars))
        except Exception:
            out[name] = np.nan

    return pd.Series(out)
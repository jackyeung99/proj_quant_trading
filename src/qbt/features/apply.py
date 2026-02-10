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
    bucket: pd.Series,
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
    feat_series: list[pd.Series] = []
    cols = set(bars.columns)

    for feat in features:
        name = str(feat.get("name"))
        requires = tuple(feat.get("requires", ()))  # type: ignore
        func = feat.get("func")

        if not name or func is None:
            continue

        # If required cols missing, return NaN for all buckets
        if any(c not in cols for c in requires):
            s = pd.Series(index=pd.Index(bucket.unique()).sort_values(), dtype="float64", name=name)
            s.loc[:] = np.nan
            feat_series.append(s)
            continue

        # Reduce intraday -> scalar per bucket
        def _safe_reduce(g: pd.DataFrame) -> float:
            try:
                return float(func(g))
            except Exception:
                return np.nan

        s = bars.groupby(bucket, sort=True).apply(_safe_reduce)
        s.name = name
        feat_series.append(s)

    feats_df = pd.concat(feat_series, axis=1) if feat_series else pd.DataFrame(index=pd.Index(bucket.unique()).sort_values())

    return feats_df

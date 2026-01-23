from __future__ import annotations
import pandas as pd
from qbt.core.types import RunSpec
from qbt.strategies.base import Strategy


class BuyHoldStrategy(Strategy):
    """
    Walk-forward compatible buy & hold.
    fit() is a no-op.
    compute_weight() always returns 1.0 for the provided slice.
    """

    def fit(self, train: pd.DataFrame, spec: RunSpec) -> None:
        # nothing to estimate
        return

    def compute_weight(self, data: pd.DataFrame, spec: RunSpec) -> pd.Series:
        return pd.Series(1.0, index=data.index, name="weight")

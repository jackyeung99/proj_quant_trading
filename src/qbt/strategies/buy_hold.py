from __future__ import annotations
import pandas as pd
from qbt.core.types import RunSpec
from qbt.strategies.base import Strategy

class BuyHoldStrategy(Strategy):
    def compute_weight(self, data: pd.DataFrame, spec: RunSpec) -> pd.Series:
        # Always fully invested
        return pd.Series(1.0, index=data.index, name="weight")

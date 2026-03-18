from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import pandas as pd

from qbt.core.types import RunSpec, ModelInputs


class Strategy(ABC):
    """
    Produces signals/forecasts
    Portfolio construction is handled elsewhere (Allocator).
    """

    @abstractmethod
    def required_asset_features(self, spec: RunSpec) -> List[str]:
        """
        Per-asset feature names needed by the model, excluding returns.

        Examples:
          ["rvol_20"]
          ["mom_60", "rv_5d"]

        These should correspond to columns in storage like:
          XLE_rvol_20, XLK_rvol_20, ...
        """
        raise NotImplementedError

    def required_global_features(self, spec: RunSpec) -> List[str]:
        """
        Shared/global feature names needed by the model.

        Examples:
          ["OVX"]
          ["DGS10", "TBILL_3M"]

        These should correspond to non-asset-specific columns in storage.
        Default is no global features.
        """
        return []

    @abstractmethod
    def fit(self, inputs: ModelInputs, spec: RunSpec) -> None:
        """Fit model on training window."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: ModelInputs, spec: RunSpec) -> pd.Series | pd.DataFrame:
        """
        Produce signals aligned to features.index.

        Return:
          - Series: single-asset signal
          - DataFrame: multi-asset signals (columns = assets)
        """
        raise NotImplementedError

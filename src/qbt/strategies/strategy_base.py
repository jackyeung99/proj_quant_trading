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
    def required_features(self, spec: RunSpec) -> List[str]:
        """Feature columns needed by the model (not returns)."""
        raise NotImplementedError

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

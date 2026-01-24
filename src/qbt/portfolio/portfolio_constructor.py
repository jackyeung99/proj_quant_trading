from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
from qbt.core.types import RunSpec


class PortfolioConstructor(ABC):
    """
    Converts signals into portfolio weights.
    """

    @abstractmethod
    def allocate(
        self,
        signals: pd.Series | pd.DataFrame,
        spec: RunSpec,
        assets: list[str],
    ) -> pd.DataFrame:
        """
        Return weights DataFrame [T x N], columns=assets, index=time.
        """
        raise NotImplementedError

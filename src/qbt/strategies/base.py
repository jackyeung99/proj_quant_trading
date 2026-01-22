from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from qbt.core.types import RunSpec

class Strategy(ABC):
    @abstractmethod
    def compute_weight(self, data: pd.DataFrame, spec: RunSpec) -> pd.Series:
        """Return target weight series indexed like data."""
        raise NotImplementedError

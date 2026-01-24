from dataclasses import dataclass
from typing import Protocol, Sequence, Optional
import pandas as pd

from qbt.core.exceptions import InvalidRunSpec, DataError
from qbt.core.types import RunSpec, ModelInputs

class DataAdapter(Protocol):
    def load(self, spec: "RunSpec") -> pd.DataFrame: ...
    def prepare(self, df: pd.DataFrame, spec: "RunSpec", required_cols: Sequence[str]) -> pd.DataFrame: ...

@dataclass
class DefaultDataAdapter:
    def load(self, spec: "RunSpec") -> pd.DataFrame:
        path = spec.data_path
        if path.endswith(".csv"):
            return pd.read_csv(path, index_col=0, parse_dates=True)
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        raise InvalidRunSpec(f"Unsupported data_path: {path}")

    def prepare(self, raw: pd.DataFrame, spec: RunSpec, required_cols: list[str]) -> ModelInputs:
        df = raw.sort_index()

        # --- returns matrix (ONLY) ---
        # Example: if your returns are stored as columns named like "ret_XLE"
        ret_cols = [f"ret_{a}" for a in spec.assets]
        ret_cols = spec.assets
        ret = df[ret_cols].div(100)
        # .rename(columns={f"ret_{a}": a for a in spec.assets})

        # --- features (ONLY what the strategy asked for) ---
        # required_cols should NOT include return columns
        features = df[required_cols] if required_cols else df.iloc[:, 0:0]

        # drop rows with missing returns (simulation contract)
        ret = ret.dropna(how="any")
        features = features.reindex(ret.index)

        return ModelInputs(ret=ret, features=features)
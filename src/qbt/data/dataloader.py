from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qbt.core.types import ModelInputs, RunSpec


class DataAdapter:
    def load(self, spec: RunSpec) -> pd.DataFrame: ...
    def prepare(self, raw: pd.DataFrame, spec: RunSpec, required_cols: list[str]) -> ModelInputs: ...


@dataclass
class DefaultDataAdapter(DataAdapter):
    """
    Assumes the input is a persisted LONG modeling table (gold) with columns:
      - date or timestamp
      - asset
      - return column (ret_oo preferred, else ret_cc)
      - feature columns

    load()   -> returns the LONG modeling table (DataFrame)
    prepare() -> returns ModelInputs with wide ret + wide features
    """

    def load(self, spec: RunSpec) -> pd.DataFrame:
        data_path = spec.data_path

        if not data_path:
            raise ValueError("spec.data must include 'data_path' pointing to the modeling table parquet/csv.")

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Modeling table not found at: {path}")


        # infer from suffix
        if path.suffix.lower() in [".parquet"]:
            fmt = "parquet"
        elif path.suffix.lower() in [".csv"]:
            fmt = "csv"
        else:
            raise ValueError(f"Could not infer file format from suffix: {path.suffix}")

        # minimal read
        if fmt == "parquet":
            df = pd.read_parquet(path)
        elif fmt == "csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file_format={fmt!r}. Use 'parquet' or 'csv'.")


        return df

    def prepare(self, raw: pd.DataFrame, spec: RunSpec, required_cols: list[str]) -> ModelInputs:
        if not isinstance(raw.columns, pd.MultiIndex):
            raise ValueError("Expected raw to have MultiIndex columns: (asset, field).")

        # pick return stream
        fields = set(raw.columns.get_level_values(1))
        ret_field = "ret_oo" if "ret_oo" in fields else "ret_cc"
        if ret_field not in fields:
            raise ValueError("Missing return field: expected 'ret_oo' or 'ret_cc' in raw columns.")

        # wide returns (date x assets)
        ret_wide = raw.xs(ret_field, level=1, axis=1).sort_index().sort_index(axis=1)

        # flatten ALL columns into "<asset>_<field>"
        X = raw.copy().sort_index().sort_index(axis=1, level=[0, 1])
        X.columns = [f"{a}_{f}" for a, f in X.columns]

        # drop return columns from features (optional but usually desired)
        drop_cols = [c for c in X.columns if c.endswith("_ret_cc") or c.endswith("_ret_oo")]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        # required columns check (expects flattened names)
        if required_cols:
            missing = [c for c in required_cols if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing required feature columns after flattening: {missing}\n"
                    f"Available (sample): {list(X.columns)[:20]}"
                )

        # align
        idx = ret_wide.index.intersection(X.index)
        ret_wide = ret_wide.loc[idx].dropna(how="all")
        X = X.loc[ret_wide.index]

        return ModelInputs(
            ret=ret_wide,
            features=X,
        )
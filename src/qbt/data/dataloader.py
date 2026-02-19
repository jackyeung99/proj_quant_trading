from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

    def prepare(
        self,
        raw: pd.DataFrame,
        spec: "RunSpec",
        required_cols: list[str],
        *,
        time_col: Optional[str] = None,          # e.g. "session_date" or "timestamp"
        asset_col: Optional[str] = None,         # e.g. "ticker" or "symbol"
        ret_candidates: Sequence[str] = ("ret_oo", "ret_cc"),
        assume_naive_tz: str = "UTC",
        drop_value_cols: Sequence[str] = (),     # extra columns to exclude from features
    ) -> "ModelInputs":
        """
        raw: LONG gold table.

        Output:
        - ret: wide (index=time_col, columns=assets)
        - features: wide flattened (index=time_col, columns="<asset>_<feature>")
        """
        df = raw.copy()

        # ----------------------------
        # 0) Resolve column names
        # ----------------------------
        if time_col is None:
            time_col = next((c for c in ("session_date", "timestamp", "date", "time") if c in df.columns), None)
        if asset_col is None:
            asset_col = next((c for c in ("ticker", "symbol", "asset", "secid") if c in df.columns), None)

        # If no time column, allow DatetimeIndex
        if time_col is None:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    "Expected a time column (e.g. 'session_date' or 'timestamp') "
                    "or a DatetimeIndex."
                )
            # create a consistent internal time column for melt/pivot
            time_col = "__time__"
            df = df.copy()
            df[time_col] = df.index

        if asset_col is None:
            raise ValueError(
                "Expected an asset column (e.g. 'ticker'/'symbol'). "
                f"Columns seen: {list(df.columns)}"
            )

        # ----------------------------
        # 1) Normalize time column
        # ----------------------------
        # If it's already datetime-like, keep it; otherwise parse.
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])

        # If tz-naive, localize to assume_naive_tz (commonly UTC).
        # If tz-aware, convert to UTC for consistency.
        if isinstance(df[time_col].dtype, pd.DatetimeTZDtype):
            df[time_col] = df[time_col].dt.tz_convert("UTC")
        else:
            df[time_col] = df[time_col].dt.tz_localize(assume_naive_tz)

        # Use as index
        df = df.sort_values([asset_col, time_col]).set_index(time_col)

        # asset dtype
        df[asset_col] = df[asset_col].astype("string")

        # ----------------------------
        # 2) Pivot long -> wide (asset, field)
        # ----------------------------
        value_cols = [c for c in df.columns if c != asset_col]
        if not value_cols:
            raise ValueError(f"No value columns found to pivot (everything except '{asset_col}').")

        wide = (
            df.reset_index()
            .melt(
                id_vars=[time_col, asset_col],
                value_vars=value_cols,
                var_name="field",
                value_name="value",
            )
            .pivot_table(
                index=time_col,
                columns=[asset_col, "field"],
                values="value",
                aggfunc="last",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        # ----------------------------
        # 3) Pick return stream (from candidates)
        # ----------------------------
        fields = set(wide.columns.get_level_values(1))
        ret_field = next((f for f in ret_candidates if f in fields), None)
        if ret_field is None:
            raise ValueError(
                f"Missing return field: expected one of {list(ret_candidates)}. "
                f"Fields available (sample): {sorted(list(fields))[:20]}"
            )

        ret_wide = wide.xs(ret_field, level=1, axis=1).sort_index().sort_index(axis=1)

        # ----------------------------
        # 4) Features: flatten "<asset>_<field>"
        # ----------------------------
        X = wide.copy().sort_index().sort_index(axis=1, level=[0, 1])
        X.columns = [f"{a}_{f}" for a, f in X.columns]

        # drop all return candidates + any extra requested drops
        drop_suffixes = tuple(f"_{f}" for f in ret_candidates)
        drop_cols = [c for c in X.columns if c.endswith(drop_suffixes)]
        drop_cols += [c for c in drop_value_cols if c in X.columns]
        if drop_cols:
            X = X.drop(columns=sorted(set(drop_cols)))

        # ----------------------------
        # 5) Required columns check (flattened names)
        # ----------------------------
        if required_cols:
            missing = [c for c in required_cols if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing required feature columns after flattening: {missing}\n"
                    f"Available (sample): {list(X.columns)[:20]}"
                )

        # ----------------------------
        # 6) Align
        # ----------------------------
        idx = ret_wide.index.intersection(X.index)
        ret_wide = ret_wide.loc[idx].dropna(how="all")
        X = X.loc[ret_wide.index]

        return ModelInputs(ret=ret_wide, features=X)
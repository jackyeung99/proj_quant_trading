from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from qbt.core.types import ModelInputs, RunSpec
from qbt.storage.storage import Storage


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
    def __init__(self, storage: Storage):
        self.storage = storage

    def load(self, spec: RunSpec) -> pd.DataFrame:
        data_path = spec.data_path
        if not data_path:
            raise ValueError("spec.data_path must point to modeling table (key for S3, or local path).")

        # If your pipeline uses Storage, treat data_path as a storage key (relative)
        key = str(data_path).lstrip("/")

        # Use storage exists/read (works for local and s3 backends)
        if not self.storage.exists(key):
            raise FileNotFoundError(f"Modeling table not found in storage at key: {key}")

        # infer format from suffix
        suffix = Path(key).suffix.lower()
        if suffix == ".parquet":
            return self.storage.read_parquet(key)
        if suffix == ".csv":
            return self.storage.read_csv(key)  # implement if you need it

        raise ValueError(f"Could not infer file format from suffix: {suffix}")

    # def prepare(
    #     self,
    #     raw: pd.DataFrame,
    #     spec: "RunSpec",
    #     required_cols: list[str],
    #     *,
    #     time_col: Optional[str] = None,          # e.g. "session_date" or "timestamp"
    #     ret_candidates: Sequence[str] = ("ret_oo", "ret_cc"),
    #     assume_naive_tz: str = "UTC",
    #     drop_value_cols: Sequence[str] = (),     # extra columns to exclude from features
    # ) -> "ModelInputs":
    #     """
    #     raw: WIDE gold table.

    #     Accepted wide schemas:
    #     A) MultiIndex columns: (asset, field)
    #         - Example columns: ("XLE", "rv"), ("XLE", "ret_cc"), ("SPY", "rv"), ...
    #         - Index: DatetimeIndex (session_date)

    #     B) Flattened columns: "<asset>_<field>"
    #         - Example columns: "XLE_rv", "XLE_ret_cc", "SPY_rv", ... 
    #         - Index: DatetimeIndex (session_date)

    #     Output:
    #     - ret: wide (index=time, columns=assets)
    #     - features: wide flattened (index=time, columns="<asset>_<feature>")
    #     """
    #     df = raw.copy()

    #     # ----------------------------
    #     # 0) Resolve / normalize time index
    #     # ----------------------------
    #     if time_col is None:
    #         # prefer index if datetime
    #         if isinstance(df.index, pd.DatetimeIndex):
    #             pass
    #         else:
    #             # otherwise look for a time column
    #             time_col = next((c for c in ("session_date", "timestamp", "date", "time") if c in df.columns), None)

    #     if time_col is not None:
    #         df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    #         df = df.dropna(subset=[time_col]).set_index(time_col)

    #     if not isinstance(df.index, pd.DatetimeIndex):
    #         raise ValueError("Expected a DatetimeIndex or a time column like 'session_date'/'timestamp'.")

    #     # timezone normalize
    #     if df.index.tz is None:
    #         df.index = df.index.tz_localize(assume_naive_tz)
    #     else:
    #         df.index = df.index.tz_convert("UTC")

    #     df = df.sort_index()
        
    #     # ----------------------------
    #     # 1) Coerce to canonical wide: MultiIndex (asset, field)
    #     # ----------------------------
    #     if isinstance(df.columns, pd.MultiIndex):
    #         if df.columns.nlevels != 2:
    #             raise ValueError(f"Expected 2-level MultiIndex columns (asset, field). Got nlevels={df.columns.nlevels}")
    #         wide = df.copy()
    #         wide.columns = pd.MultiIndex.from_tuples([(str(a), str(f)) for a, f in wide.columns], names=["asset", "field"])
    #     else:
    #         # Flattened "<asset>_<field>" → MultiIndex
    #         cols = [str(c) for c in df.columns]
    #         pairs = []
    #         bad = []
    #         for c in cols:
    #             if "_" not in c:
    #                 bad.append(c)
    #                 continue
    #             a, f = c.split("_", 1)
    #             pairs.append((a, f))
    #         if bad:
    #             raise ValueError(
    #                 "Wide table has non-MultiIndex columns but some columns are not '<asset>_<field>'. "
    #                 f"Examples: {bad[:10]}"
    #             )
    #         wide = df.copy()
    #         wide.columns = pd.MultiIndex.from_tuples(pairs, names=["asset", "field"])

    #     wide = wide.sort_index().sort_index(axis=1)

    #     # ----------------------------
    #     # 2) Pick return stream from candidates
    #     # ----------------------------
    #     fields = set(wide.columns.get_level_values("field"))
    #     ret_field = next((f for f in ret_candidates if f in fields), None)
    #     if ret_field is None:
    #         raise ValueError(
    #             f"Missing return field: expected one of {list(ret_candidates)}. "
    #             f"Fields available (sample): {sorted(list(fields))[:30]}"
    #         )

    #     ret_wide = wide.xs(ret_field, level="field", axis=1).sort_index(axis=1)

    #     # ----------------------------
    #     # 3) Features: flatten "<asset>_<field>" and drop returns
    #     # ----------------------------
    #     X = wide.copy()
    #     X.columns = [f"{a}_{f}" for a, f in X.columns]

    #     # drop all return candidates + any extra requested drops (exact names)
    #     drop_suffixes = tuple(f"_{f}" for f in ret_candidates)
    #     drop_cols = [c for c in X.columns if c.endswith(drop_suffixes)]
    #     drop_cols += [c for c in drop_value_cols if c in X.columns]
    #     if drop_cols:
    #         X = X.drop(columns=sorted(set(drop_cols)))

    #     # ----------------------------
    #     # 4) Required columns check
    #     # ----------------------------
    #     if required_cols:
    #         missing = [c for c in required_cols if c not in X.columns]
    #         if missing:
    #             raise ValueError(
    #                 f"Missing required feature columns: {missing}\n"
    #                 f"Available (sample): {list(X.columns)[:30]}"
    #             )

    #     # ----------------------------
    #     # 5) Align + basic cleanup
    #     # ----------------------------
    #     idx = ret_wide.index.intersection(X.index)
    #     ret_wide = ret_wide.loc[idx].dropna(how="all")
    #     X = X.loc[ret_wide.index]

    #     return ModelInputs(ret=ret_wide, features=X)


    def prepare(
        self,
        raw: pd.DataFrame,
        spec: "RunSpec",
        required_cols: list[str],
        *,
        time_col: Optional[str] = None,
        ret_candidates: Sequence[str] = ("XLE_ret_oo", "XLE_ret_cc"),
        assume_naive_tz: str = "UTC",
        drop_value_cols: Sequence[str] = (),
    ) -> "ModelInputs":

        df = raw.copy()

        # ----------------------------
        # 1) Time handling
        # ----------------------------
        if time_col is None:
            if not isinstance(df.index, pd.DatetimeIndex):
                time_col = next(
                    (c for c in ("session_date", "timestamp", "date", "time") if c in df.columns),
                    None,
                )

        if time_col is not None:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).set_index(time_col)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Expected DatetimeIndex or valid time column.")

        # timezone normalize
        if df.index.tz is None:
            df.index = df.index.tz_localize(assume_naive_tz)
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index()

        # ----------------------------
        # 2) Extract returns
        # ----------------------------
        ret_col = next((c for c in ret_candidates if c in df.columns), None)
        if ret_col is None:
            raise ValueError(f"No return column found among {ret_candidates}")

        asset = ret_col.split("_")[0]

        ret_wide = df[[ret_col]].rename(columns={ret_col: asset})

        # ----------------------------
        # 3) Features
        # ----------------------------
        X = df.drop(columns=[ret_col], errors="ignore")

        if drop_value_cols:
            X = X.drop(columns=list(drop_value_cols), errors="ignore")

        # required feature check
        if required_cols:
            missing = [c for c in required_cols if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing required feature columns: {missing}\n"
                    f"Available columns: {list(X.columns)[:30]}"
                )

        # ----------------------------
        # 4) Align
        # ----------------------------
        idx = ret_wide.index.intersection(X.index)
        ret_wide = ret_wide.loc[idx].dropna(how="all")
        X = X.loc[ret_wide.index]

        return ModelInputs(ret=ret_wide, features=X)
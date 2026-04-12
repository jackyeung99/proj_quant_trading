from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from qbt.core.types import ModelInputs
from qbt.config.specs import StrategySpec
from qbt.storage.storage import Storage
from qbt.storage.paths import StoragePaths


class DataAdapter:
    def load(self, spec: StrategySpec) -> pd.DataFrame: ...
    def prepare(
        self,
        raw: pd.DataFrame,
        spec: StrategySpec,
        required_asset_features: list[str],
    ) -> ModelInputs: ...


@dataclass
class WidePrefixDataAdapter(DataAdapter):
    """
    Assumes a wide modeling table with columns like:
      - session_date / timestamp / date
      - XLE_ret_cc, XLK_ret_cc, ...
      - XLE_rvol, XLK_rvol, ...
      - OVX, DGS10, TBILL_3M, ...   # global features

    Responsibilities:
      1. Resolve and load the wide gold/model table from storage using spec.input_table
      2. Adapt the raw table into ModelInputs
    """
    storage: Storage
    paths: StoragePaths
    default_time_candidates: Sequence[str] = ("session_date", "timestamp", "date", "time")
    default_ret_suffixes: Sequence[str] = ("ret_cc",)
    assume_naive_tz: str = "UTC"

    def load(self, spec: StrategySpec) -> pd.DataFrame:
        input_table = spec.input_table
        if not input_table:
            raise ValueError("strategy.input_table must be set.")

        key = self.paths.gold_table_key(freq=spec.input_table_freq, tag=input_table)

        if not self.storage.exists(key):
            raise FileNotFoundError(
                f"Model table not found for input_table='{input_table}': {key}"
            )

        return self.storage.read_parquet(key)

    def prepare(
        self,
        raw: pd.DataFrame,
        spec: StrategySpec,
        required_asset_features: Sequence[str],
        required_global_features: Sequence[str] = (),
        *,
        time_col: Optional[str] = None,
        ret_suffixes: Optional[Sequence[str]] = None,
        drop_value_cols: Sequence[str] = (),
        assets: Optional[Sequence[str]] = None,
    ) -> ModelInputs:
        df = raw.copy()
        df = self._normalize_time_index(df, time_col=time_col)

        resolved_assets = self._resolve_assets(spec=spec, assets=assets)

        ret_wide, ret_cols = self._extract_returns_wide(
            df,
            ret_suffixes=ret_suffixes or self.default_ret_suffixes,
            assets=resolved_assets,
        )

        asset_features, used_asset_feature_cols = self._extract_asset_features(
            df,
            required_asset_features=required_asset_features,
            assets=list(ret_wide.columns),
        )

        global_features = self._extract_global_features(
            df,
            required_global_features=required_global_features,
            exclude_cols=set(ret_cols) | set(used_asset_feature_cols) | set(drop_value_cols),
        )

        ret_wide, asset_features, global_features = self._align_all(
            ret_wide=ret_wide,
            asset_features=asset_features,
            global_features=global_features,
        )

        return ModelInputs(
            ret=ret_wide,
            asset_features=asset_features,
            global_features=global_features,
        )

    def _resolve_assets(
        self,
        *,
        spec: StrategySpec,
        assets: Optional[Sequence[str]],
    ) -> Optional[list[str]]:
        if assets is not None:
            resolved = list(assets)
        else:
            resolved = list(spec.assets or [])

        resolved = [a for a in resolved if a is not None and str(a) != ""]
        if not resolved:
            return None

        return resolved

    def _normalize_time_index(
        self,
        df: pd.DataFrame,
        *,
        time_col: Optional[str],
    ) -> pd.DataFrame:
        out = df.copy()

        if time_col is None and not isinstance(out.index, pd.DatetimeIndex):
            time_col = next((c for c in self.default_time_candidates if c in out.columns), None)

        if time_col is not None:
            out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
            out = out.dropna(subset=[time_col]).set_index(time_col)

        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("Expected DatetimeIndex or a valid time column.")

        if out.index.tz is None:
            out.index = out.index.tz_localize(self.assume_naive_tz)
        else:
            out.index = out.index.tz_convert("UTC")

        return out.sort_index()

    def _extract_returns_wide(
        self,
        df: pd.DataFrame,
        *,
        ret_suffixes: Sequence[str],
        assets: Optional[Sequence[str]] = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        suffixes = tuple(ret_suffixes)
        ret_cols = [c for c in df.columns if c.endswith(suffixes)]

        if not ret_cols:
            raise ValueError(
                f"No return columns found using suffixes={list(ret_suffixes)}. "
                f"Sample columns: {list(df.columns)[:20]}"
            )

        ret_wide = df[ret_cols].rename(columns=self._asset_from_prefixed_col)

        if ret_wide.columns.duplicated().any():
            dupes = ret_wide.columns[ret_wide.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate asset names after return rename: {dupes}")

        if assets is not None:
            requested_assets = list(assets)
            missing_assets = [a for a in requested_assets if a not in ret_wide.columns]
            if missing_assets:
                raise ValueError(f"Missing requested return columns for assets: {missing_assets}")
            ret_wide = ret_wide.loc[:, requested_assets]

        ret_cols_used = [
            c for c in ret_cols
            if self._asset_from_prefixed_col(c) in ret_wide.columns
        ]

        return ret_wide, ret_cols_used

    def _extract_asset_features(
        self,
        df: pd.DataFrame,
        *,
        required_asset_features: Sequence[str],
        assets: Sequence[str],
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        feature_panels: dict[str, pd.DataFrame] = {}
        used_cols: list[str] = []

        for feat in required_asset_features:
            cols = []
            rename_map = {}

            for asset in assets:
                col = f"{asset}_{feat}"
                if col not in df.columns:
                    raise ValueError(
                        f"Missing required asset feature '{feat}' for asset '{asset}'. "
                        f"Expected column: {col}"
                    )
                cols.append(col)
                rename_map[col] = asset

            panel = df.loc[:, cols].rename(columns=rename_map)

            if panel.columns.duplicated().any():
                dupes = panel.columns[panel.columns.duplicated()].tolist()
                raise ValueError(f"Duplicate asset names in feature panel '{feat}': {dupes}")

            feature_panels[feat] = panel
            used_cols.extend(cols)

        return feature_panels, used_cols

    def _extract_global_features(
        self,
        df: pd.DataFrame,
        *,
        required_global_features: Sequence[str],
        exclude_cols: set[str],
    ) -> pd.DataFrame:
        missing = [c for c in required_global_features if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required global features: {missing}\n"
                f"Available columns (sample): {list(df.columns)[:30]}"
            )

        if required_global_features:
            return df.loc[:, list(required_global_features)].copy()

        global_cols = [
            c for c in df.columns
            if c not in exclude_cols and "_" not in c
        ]
        return df.loc[:, global_cols].copy()

    def _align_all(
        self,
        *,
        ret_wide: pd.DataFrame,
        asset_features: dict[str, pd.DataFrame],
        global_features: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
        ret_wide = ret_wide.dropna(how="all")
        idx = ret_wide.index

        aligned_asset_features = {
            feat: panel.loc[idx]
            for feat, panel in asset_features.items()
        }

        global_features = global_features.loc[idx]

        return ret_wide, aligned_asset_features, global_features

    @staticmethod
    def _asset_from_prefixed_col(col: str) -> str:
        return col.split("_", 1)[0]
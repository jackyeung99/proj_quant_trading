from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import pandas as pd

from qbt.core.types import RunSpec, ModelInputs
from qbt.strategies.strategy_base import Strategy
from qbt.strategies.strategy_registry import register_strategy


from qbt.strategies.state_model import StateSignalModel


@register_strategy("MultiAssetStateSignal")
class MultiStateSignalModel(Strategy):
    """
    Multi-asset wrapper around multiple StateSignalModel instances.

    Behavior:
      - fits one StateSignalModel per asset
      - each child model sees a single-asset ModelInputs slice
      - explicitly sets child w_high = 1 / m where m = # assets
      - returns a wide weight DataFrame [time x asset]

    Persistence:
      - aggregates each child model's persisted series with asset-prefixed columns
      - adds portfolio-level summaries:
          active_count
          target_gross
    """

    def __init__(self) -> None:
        self.models_: dict[str, StateSignalModel] = {}
        self.assets_: list[str] = []

    def required_asset_features(self, spec: RunSpec) -> list[str]:
        # same requirement as the single-asset model
        params = spec.params or {}
        state_source = params.get("state_source", "asset")
        if state_source == "global":
            return []
        state_var = params.get("state_var")
        if state_var is None:
            raise ValueError("MultiStateSignalModel requires params['state_var'].")
        return [str(state_var)]

    def required_global_features(self, spec: RunSpec) -> list[str]:
        params = spec.params or {}
        state_source = params.get("state_source", "asset")
        if state_source != "global":
            return []
        state_var = params.get("state_var")
        if state_var is None:
            raise ValueError("MultiStateSignalModel requires params['state_var'].")
        return [str(state_var)]


    def _child_spec(self, spec: RunSpec, *, m: int) -> RunSpec:
        params = dict(spec.params or {})
        params["w_min"] = 0.0
        params["w_high"] = 1.0 / float(m)

        return replace(spec, params=params)

    def _slice_single_asset_inputs(
        self,
        inputs: ModelInputs,
        *,
        asset: str,
    ) -> ModelInputs:
        return ModelInputs(
            ret=inputs.ret.loc[:, [asset]],
            asset_features={
                name: panel.loc[:, [asset]]
                for name, panel in inputs.asset_features.items()
                if asset in panel.columns
            },
            global_features=inputs.global_features.copy(),
        )

    def fit(self, inputs: ModelInputs, spec: RunSpec) -> None:
        assets = list(inputs.ret.columns)
        if not assets:
            raise ValueError("MultiStateSignalModel.fit received no assets.")

        m = len(assets)
        child_spec = self._child_spec(spec, m=m)

        self.assets_ = assets
        self.models_ = {}

        for asset in assets:
            child_inputs = self._slice_single_asset_inputs(inputs, asset=asset)
            model = StateSignalModel()
            model.fit(child_inputs, child_spec)
            self.models_[asset] = model

    def predict(self, inputs: ModelInputs, spec: RunSpec) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("predict called before fit().")

        assets = list(inputs.ret.columns)
        m = len(assets)
        child_spec = self._child_spec(spec, m=m)

        out: dict[str, pd.Series] = {}

        for asset in assets:
            if asset not in self.models_:
                raise ValueError(f"No fitted child model found for asset '{asset}'.")

            child_inputs = self._slice_single_asset_inputs(inputs, asset=asset)
            child_w = self.models_[asset].predict(child_inputs, child_spec)

            if isinstance(child_w, pd.DataFrame):
                if child_w.shape[1] != 1:
                    raise ValueError(
                        f"Child model for asset '{asset}' returned unexpected shape {child_w.shape}."
                    )
                out[asset] = child_w.iloc[:, 0]
            else:
                out[asset] = child_w

        w = pd.DataFrame(out, index=inputs.ret.index).reindex(columns=assets)
        return w.fillna(0.0)

    def get_model_state(self) -> dict:
        return {
            "assets": self.assets_,
            "n_assets": len(self.assets_),
            "children": {
                asset: model.get_model_state()
                for asset, model in self.models_.items()
            },
        }

    def get_persisted_series(self, *, test_inputs: ModelInputs, spec: RunSpec) -> pd.DataFrame:
        if not self.models_:
            return pd.DataFrame(index=test_inputs.ret.index)

        assets = list(test_inputs.ret.columns)
        m = len(assets)
        child_spec = self._child_spec(spec, m=m)

        blocks: list[pd.DataFrame] = []
        signal_cols: list[str] = []

        for asset in assets:
            model = self.models_.get(asset)
            if model is None:
                continue

            child_inputs = self._slice_single_asset_inputs(test_inputs, asset=asset)
            child_extra = model.get_persisted_series(test_inputs=child_inputs, spec=child_spec)

            if child_extra is None or child_extra.empty:
                continue

            child_extra = child_extra.reindex(test_inputs.ret.index)
            child_extra = child_extra.add_prefix(f"{asset}__")
            blocks.append(child_extra)

            sig_col = f"{asset}__signal"
            if sig_col in child_extra.columns:
                signal_cols.append(sig_col)

        if blocks:
            out = pd.concat(blocks, axis=1)
        else:
            out = pd.DataFrame(index=test_inputs.ret.index)

        if signal_cols:
            sig_df = out[signal_cols].fillna(0.0)
            out["active_count"] = sig_df.sum(axis=1)
            out["target_gross"] = sig_df.sum(axis=1) * (1.0 / float(m))
        else:
            out["active_count"] = 0.0
            out["target_gross"] = 0.0

        return out
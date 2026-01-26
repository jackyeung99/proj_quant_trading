from qbt.strategies.base import Strategy
from qbt.strategies.registry import register_strategy
from qbt.core.types import RunSpec, ModelInputs
import pandas as pd
import numpy as np

@register_strategy("BuyHold")
class BuyHoldStrategy(Strategy):

    def required_features(self, spec):
        return []  # no features needed
    
    def predict(self, inputs, spec):
        return super().predict(inputs, spec)

    def fit(self, inputs: ModelInputs, spec: RunSpec):
        pass  # nothing to fit

    def predict(self, inputs: ModelInputs, spec: RunSpec) -> pd.DataFrame:
        # for compatibility: treat prediction as weights
        return self.compute_weight(inputs, spec)
    
    def compute_weight(self, inputs: ModelInputs, spec: RunSpec) -> pd.DataFrame:
        R = inputs.ret.sort_index()
        assets = list(R.columns)

        if len(assets) == 0:
            raise ValueError("inputs.ret has no asset columns.")

        params = spec.params or {}
        normalize = bool(params.get("normalize", True))

        # 1) If user provided explicit weights, use them (missing tickers -> 0)
        w_map = params.get("buyhold_weights")
        if isinstance(w_map, dict) and len(w_map) > 0:
            w0 = pd.Series({a: float(w_map.get(a, 0.0)) for a in assets}, dtype=float)
        else:
            # 2) Default equal weights across all assets
            w0 = pd.Series(1.0 / len(assets), index=assets, dtype=float)

        if normalize:
            s = float(w0.abs().sum())
            if s > 0:
                w0 = w0 / s

        # constant weights over time
        W = pd.DataFrame(np.tile(w0.values, (len(R.index), 1)), index=R.index, columns=assets)
        return W

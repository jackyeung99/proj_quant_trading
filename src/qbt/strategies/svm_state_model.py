from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qbt.core.types import RunSpec, ModelInputs
from qbt.strategies.strategy_base import Strategy
from qbt.strategies.strategy_registry import register_strategy
from qbt.core.logging import get_logger

logging = get_logger(__name__)

@register_strategy("SVMStateSignal")
class SVMStateSignalModel(Strategy):
    def __init__(self) -> None:
        self.svm_model = None
        self.features_: list[str] = []
        self.w_low_: float = 0.0
        self.w_high_: float = 0.0
        self.lag_state_: int = 1
        self.state_vars: list[str] = []

    def parse_params(self, spec: RunSpec) -> dict:
        params = spec.params or {}

        # Accept either "features" or old "state_var"
        features = params.get("features", None)
        if features is None:
            features = params.get("state_var", None)

        if not features:
            raise ValueError("SVMStateSignal requires params['features'] (or 'state_var').")
        if isinstance(features, str):
            features = [features]
        if not isinstance(features, (list, tuple)):
            raise ValueError("SVMStateSignal requires params['features'] as a list (or string).")

        return {
            "features": list(features),
            # threshold is a quantile (0..1) in your code
            "threshold": float(params.get("return_threshold", 0.7)),
            "lag_state": int(params.get("lag_state", 1)),
            "gamma": float(params.get("gamma", 5.0)),
            "w_min": float(params.get("w_min", 0.0)),
            "w_max": float(params.get("w_max", 3.0)),
            "eps": float(params.get("eps", 1e-12)),
            # SVM hyperparams in YAML under params.svm_params (optional)
            "svm_params": dict(params.get("svm_params", {})),
        }

    def required_features(self, spec: RunSpec) -> list[str]:
        return self.parse_params(spec)["features"]

    def fit(self, inputs: ModelInputs, spec: RunSpec) -> None:
        p = self.parse_params(spec)
        self.state_vars = p["features"]
        self.lag_state_ = p["lag_state"]

        X = inputs.features.sort_index()
        r = inputs.ret.sort_index()

        missing = [c for c in self.state_vars if c not in X.columns]
        if missing:
            raise ValueError(f"Train features missing columns: {missing}")

        r1 = r.iloc[:, 0].rename("ret")

        # lag features so features at t-1 predict ret regime at t
        S = X.loc[:, self.state_vars].shift(self.lag_state_)

        df = pd.concat([r1, S], axis=1)
        df = df.dropna(subset=["ret"] + self.state_vars)

        if len(df) < 20:
            self.w_low_, self.w_high_ = 0.0, 0.0
            self.svm_model = None
            self.features_ = []
            return

        df = self.classify_returns(df, p["threshold"])

        y = (df["regime"] == "state_high").astype(int)

        X_train = df.loc[:, self.state_vars]
        self.fit_svm(X_train, y, p["svm_params"])

        low_df = df[df["regime"] == "state_low"]
        high_df = df[df["regime"] == "state_high"]

        w_low, w_high = self.estimate_weights_meanvar(
            low_df, high_df,
            gamma=p["gamma"],
            w_min=p["w_min"],
            w_max=p["w_max"],
            eps=p["eps"],
        )
        self.w_low_ = float(w_low)
        self.w_high_ = float(w_high)

    def predict(self, inputs: ModelInputs, spec: RunSpec) -> pd.Series:
        if self.svm_model is None:
            # If a fold had too little data, just output 0 weights
            return pd.Series(0.0, index=inputs.ret.index, name="weight")

        X = inputs.features.sort_index()

        # Use same features as training, and apply the same lag timing rule
        missing = [c for c in self.features_ if c not in X.columns]
        if missing:
            raise ValueError(f"Predict features missing columns: {missing}")

        S = X.loc[:, self.features_].shift(self.lag_state_)

        # Align output to the test ret index; if shift creates NaNs, weight=0 there
        S = S.reindex(inputs.ret.index)
        mask_ok = ~S.isna().any(axis=1)

        yhat = pd.Series(0, index=S.index, dtype=int)
        if mask_ok.any():
            yhat.loc[mask_ok] = self.svm_model.predict(S.loc[mask_ok, self.features_])


        # Map predicted regime -> portfolio weight
        w = np.where(yhat.values == 1, self.w_high_, self.w_low_)
        
        return pd.Series(w, index=S.index, name="weight")

    def classify_returns(self, df: pd.DataFrame, threshold_q: float) -> pd.DataFrame:
        thr = float(np.quantile(df["ret"], threshold_q))
        thr = max(thr, 0.0)

        out = df.copy()
        out["regime"] = "state_low"
        out.loc[out["ret"] >= thr, "regime"] = "state_high"
        return out

    def fit_svm(self, X: pd.DataFrame, y: pd.Series, model_params: dict) -> None:
        self.features_ = list(X.columns)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", svm.SVC(**model_params)),
        ])
        pipe.fit(X.values, y.values)

        self.svm_model = pipe

    @staticmethod
    def estimate_weights_meanvar(
        low_regime_df: pd.DataFrame,
        high_regime_df: pd.DataFrame,
        gamma: float = 5.0,
        w_min: float = 0.0,
        w_max: float = 3.0,
        eps: float = 1e-12,
    ) -> tuple[float, float]:
        def mv_weight(x: pd.Series) -> float:
            if len(x) < 5:
                return float(np.clip(0.0, w_min, w_max))
            mu = float(x.mean())
            var = float(x.var(ddof=1))
            if not np.isfinite(var) or var < eps:
                return float(np.clip(0.0, w_min, w_max))
            w = mu / (gamma * var)
            return float(np.clip(w, w_min, w_max))

        return mv_weight(low_regime_df["ret"]), mv_weight(high_regime_df["ret"])

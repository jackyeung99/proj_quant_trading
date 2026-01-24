from __future__ import annotations

import numpy as np
import pandas as pd

from qbt.core.types import RunSpec, ModelInputs
from qbt.strategies.base import Strategy
from qbt.strategies.registry import register_strategy

from qbt.metrics.summary import _sharpe

@register_strategy("StateSignal")
class StateSignalModel(Strategy):
    """
    Learns a threshold tau that best separates Sharpe in train.
    Outputs a binary signal:
      signal[t] = 1 if S_used[t] > tau else 0
    """

    def __init__(self) -> None:
        self.tau_: float | None = None
        self.w_low_: float | None = None
        self.w_high_: float | None = None
        self.lag_state_: int = 1
        self.state_var_: str | None = None

    def parse_params(self, spec: RunSpec) -> dict:
        params = spec.params or {}

        state_var = params.get("state_var")
        if state_var is None:
            raise ValueError("StateSignalModel requires params['state_var'].")

        return {
            "state_var": state_var,
            "lag_state": int(params.get("lag_state", 1)),
            "min_frac": float(params.get("min_frac", 0.10)),
            "ann_factor": int(params.get("ann_factor", 252)),
            "gamma": float(params.get("gamma", 5.0)),
            "w_min": float(params.get("w_min", 0.0)),
            "w_max": float(params.get("w_max", 3.0)),
            "eps": float(params.get("eps", 1e-12)),
        }

    def required_columns(self, spec: RunSpec) -> list[str]:
        """
        Features needed from inputs.features (returns are always in inputs.ret).
        """
        p = self.parse_params(spec)
        return [p["state_var"]]

    def fit(self, inputs: ModelInputs, spec: RunSpec) -> None:
        p = self.parse_params(spec)
        self.state_var_ = p["state_var"]
        self.lag_state_ = p["lag_state"]

        X = inputs.features.sort_index().copy()
        r = inputs.ret.sort_index()

        if self.state_var_ not in X.columns:
            raise ValueError(f"Train features missing state_var '{self.state_var_}'")

        # single-asset assumption for this particular model:
        if not isinstance(r, pd.DataFrame) or r.shape[1] != 1:
            raise ValueError("StateSignalModel expects inputs.ret to be [T x 1] for fitting.")
        r1 = r.iloc[:, 0]

        S_used = X[self.state_var_].shift(self.lag_state_)
        df = pd.DataFrame({"S_used": S_used, "ret": r1}).dropna()

        if df.empty or len(df) < 20:
            self.tau_, self.w_low_, self.w_high_ = np.nan, 0.0, 0.0
            return

        tau = self.choose_tau_max_sep_sharpe(
            df,
            state_var="S_used",
            return_col="ret",
            min_frac=p["min_frac"],
            ann_factor=p["ann_factor"],
        )

        if tau is None or np.isnan(tau):
            self.tau_, self.w_low_, self.w_high_ = np.nan, 0.0, 0.0
            return

        w_low, w_high = self.estimate_weights_meanvar(
            df,
            state_var="S_used",
            return_col="ret",
            tau=float(tau),
            gamma=p["gamma"],
            w_min=p["w_min"],
            w_max=p["w_max"],
            eps=p["eps"],
        )

        self.tau_ = float(tau)
        self.w_low_ = float(w_low)
        self.w_high_ = float(w_high)

    def predict(self, inputs: ModelInputs, spec: RunSpec) -> pd.Series:
        if self.tau_ is None:
            raise RuntimeError("predict called before fit().")

        p = self.parse_params(spec)
        X = inputs.features.sort_index().copy()

        if p["state_var"] not in X.columns:
            raise ValueError(f"Features missing state_var '{p['state_var']}'")

        S_used = X[p["state_var"]].shift(self.lag_state_)

        # binary signal: 1=high regime, 0=low regime (NaN -> 0)
        sig = (S_used > float(self.tau_)).astype(float).fillna(0.0)
        sig.name = "state_high"
        return sig

    # ----------------------------
    # Helpers (unchanged)
    # ----------------------------

    def choose_tau_max_sep_sharpe(
        self,
        df_train: pd.DataFrame,
        state_var: str,
        return_col: str,
        min_frac: float = 0.10,
        ann_factor: int = 252,
        n_grid: int = 201,
    ) -> float | None:
        s = df_train[state_var].to_numpy()
        r = df_train[return_col].to_numpy()

        if len(s) < 20:
            return None

        qs = np.linspace(min_frac, 1 - min_frac, n_grid)
        candidates = np.unique(np.quantile(s, qs))

        best_tau = None
        best_obj = -np.inf

        for tau in candidates:
            mask = s <= tau
            n_low = int(mask.sum())
            n_high = int((~mask).sum())
            if n_low < 5 or n_high < 5:
                continue

            sr_low = _sharpe(pd.Series(r[mask]), ann_factor=ann_factor)
            sr_high = _sharpe(pd.Series(r[~mask]), ann_factor=ann_factor)
            if np.isnan(sr_low) or np.isnan(sr_high):
                continue

            obj = abs(sr_high - sr_low)
            if obj > best_obj:
                best_obj = obj
                best_tau = float(tau)

        return best_tau

    @staticmethod
    def estimate_weights_meanvar(
        df_train: pd.DataFrame,
        state_var: str,
        return_col: str,
        tau: float,
        gamma: float = 5.0,
        w_min: float = 0.0,
        w_max: float = 3.0,
        eps: float = 1e-12,
    ) -> tuple[float, float]:
        s = df_train[state_var]
        r = df_train[return_col]

        low = r[s <= tau].dropna()
        high = r[s > tau].dropna()

        def mv_weight(x: pd.Series) -> float:
            if len(x) < 5:
                return float(np.clip(0.0, w_min, w_max))
            mu = float(x.mean())
            var = float(x.var(ddof=1))
            if not np.isfinite(var) or var < eps:
                return float(np.clip(0.0, w_min, w_max))
            w = mu / (gamma * var)
            return float(np.clip(w, w_min, w_max))

        return mv_weight(low), mv_weight(high)

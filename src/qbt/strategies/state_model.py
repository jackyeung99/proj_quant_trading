from __future__ import annotations

import numpy as np
import pandas as pd

from qbt.core.types import RunSpec, ModelInputs
from qbt.strategies.strategy_base import Strategy
from qbt.strategies.strategy_registry import register_strategy

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
        self.state_var_: str | None = None

    def parse_params(self, spec: RunSpec) -> dict:
        params = spec.params or {}

        state_var = params.get("state_var")
        if state_var is None:
            raise ValueError("StateSignalModel requires params['state_var'].")

        return {
            "state_var": state_var,
            "min_frac": float(params.get("min_frac", 0.10)),
            "ann_factor": int(params.get("ann_factor", 252)),
            "gamma": float(params.get("gamma", 5.0)),
            "n_grid": int(params.get("n_grid", 100)),
            "weight_type": str(params.get('weight_allocation', 'binary')),
            "w_min": float(params.get("w_min", 0.0)),
            "w_max": float(params.get("w_high", 1.0)),
            "eps": float(params.get("eps", 1e-12)),
        }
    
    def get_model_state(self) -> dict:
        return {
            "tau_star": self.tau_,
            "w_low": self.w_low_,
            "w_high": self.w_high_,
            "state_var": self.state_var_
        }
    
    def get_persisted_series(self, *, test_inputs: ModelInputs, spec: RunSpec) -> pd.DataFrame:
        p = self.parse_params(spec)
        state_var = p["state_var"]

        X = test_inputs.features.sort_index()
        S = pd.to_numeric(X[state_var], errors="coerce")

        # what you want to plot later
        out = pd.DataFrame(
            {
                "state_value": S,
                "signal": (S < float(self.tau_)).astype(float) if self.tau_ is not None else np.nan,
                "tau_star": float(self.tau_) if self.tau_ is not None else np.nan,
                "w_low": self.w_low_,
                "w_high": self.w_high_,
            },
            index=test_inputs.ret.index,
        )

        return out

    

    def required_asset_features(self, spec: RunSpec) -> list[str]:
        p = self.parse_params(spec)
        return [p["state_var"]]

  

    def fit(self, inputs: ModelInputs, spec: RunSpec) -> None:
        p = self.parse_params(spec)
        self.state_var_ = p["state_var"]

        X = inputs.asset_features.sort_index().copy()
        r = inputs.ret.sort_index()

        if self.state_var_ not in X.columns:
            raise ValueError(f"Train features missing state_var '{self.state_var_}'")

        # single-asset assumption for this particular model:
        if not isinstance(r, pd.DataFrame) or r.shape[1] != 1:
            raise ValueError("StateSignalModel expects inputs.ret to be [T x 1] for fitting.")
        r1 = r.iloc[:, 0]

        S_used = X[self.state_var_]
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
            n_grid=p["n_grid"]
        )

        if tau is None or np.isnan(tau):
            self.tau_, self.w_low_, self.w_high_ = np.nan, 0.0, 0.0
            return

        if p['weight_type']== 'mean_var':
            w_low, w_high = self.estimate_weights_meanvar(
                df,
                state_var="S_used",
                return_col="ret",
                tau=float(tau),
                n_grid=p["n_grid"],
                gamma=p["gamma"],
                w_lo=p["w_min"],
                w_hi=p["w_max"],
                eps=p["eps"],
            )
        else:
            w_low, w_high = 1, 0

        self.tau_ = float(tau)
        self.w_low_ = float(w_low)
        self.w_high_ = float(w_high)

    def predict(self, inputs: ModelInputs, spec: RunSpec) -> pd.Series:
        if self.tau_ is None:
            raise RuntimeError("predict called before fit().")

        X = inputs.asset_features.sort_index()

        p = self.parse_params(spec)
        state_var = p["state_var"]

        if state_var not in X.columns:
            raise ValueError(f"Features missing state_var '{state_var}'")

        S_used = X[state_var]

        # 1 = high regime, 0 = low regime
        sig = (S_used > float(self.tau_)).astype(float)

        # map signal -> weights; NaN -> 0 weight
        w_vals = np.where(
            sig.fillna(0.0).values > 0.5,
            self.w_high_,
            self.w_low_,
        )


        asset = inputs.ret.columns[0]

        # build time-indexed DataFrame
        w = pd.DataFrame(
            w_vals,
            index=S_used.index,
            columns=[asset],
            dtype=float,
        )

        # align to returns index (important in walk-forward)
        w = w.reindex(inputs.ret.index).fillna(0.0)

        return w
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
        n_grid: int = 100,
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

            obj = sr_high - sr_low
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
        *,
        # grid controls
        w_lo: float = 0.0,
        w_hi: float = 1.0,
        n_grid: int = 100,
        # safety / stats controls
        min_n: int = 5,
        eps: float = 1e-12,
        ddof: int = 1,
    ) -> tuple[float, float]:
        """
        Choose (w_low, w_high) via linear scan over weights.

        Objective (mean-variance utility):
            U(w) = w*mu - 0.5*gamma*(w^2)*var

        Returns:
            (best_w_low, best_w_high)
        """
        s = pd.to_numeric(df_train[state_var], errors="coerce")
        r = pd.to_numeric(df_train[return_col], errors="coerce")

        low = r[s <= tau].dropna()
        high = r[s > tau].dropna()

        grid = np.linspace(float(w_lo), float(w_hi), int(n_grid))

        def best_w(x: pd.Series) -> float:
            if len(x) < min_n:
                return float(w_lo)

            mu = float(x.mean())
            var = float(x.var(ddof=ddof))

            # If var is tiny/invalid, utility is ~ linear in w: pick endpoint based on sign(mu)
            if (not np.isfinite(var)) or var < eps:
                return float(w_hi if mu > 0 else w_lo)

            # Vectorized utility over grid
            # U(w) = w*mu - 0.5*gamma*w^2*var
            U = grid * mu - 0.5 * float(gamma) * (grid ** 2) * var
            j = int(np.nanargmax(U))
            return float(grid[j])

        return best_w(low), best_w(high)
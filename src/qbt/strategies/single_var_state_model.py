from __future__ import annotations

import numpy as np
import pandas as pd

from qbt.core.types import RunSpec
from qbt.strategies.base import Strategy
from qbt.metrics.summary import _sharpe


class StateStrategy(Strategy):
    """
    Walk-forward compatible state timing strategy:

    fit(train):
      - choose tau maximizing Sharpe separation on train
      - estimate w_low, w_high on train (mean-variance identification)

    compute_weight(test):
      - apply regime rule using stored tau/w_low/w_high on test
      - (optionally) lag the state within the test slice
    """

    def __init__(self) -> None:
        self.tau_: float | None = None
        self.w_low_: float | None = None
        self.w_high_: float | None = None
        self.lag_state_: int = 1
        self.state_var_: str | None = None

    def __init__(self) -> None:
        self.tau_ = None
        self.w_low_ = None
        self.w_high_ = None
        self.state_var_ = None
        self.lag_state_ = 1

    # ----------------------------
    # Parameter parsing
    # ----------------------------

    def parse_params(self, spec: RunSpec) -> dict:
        params = spec.params or {}

        state_var = params.get("state_var")
        if state_var is None:
            raise ValueError("StateStrategy requires params['state_var'].")

        return_col = spec.ret_col
        if return_col is None:
            raise ValueError("RunSpec must define ret_col.")

        return {
            "state_var": state_var,
            "return_col": 'ret',
            "lag_state": int(params.get("lag_state", 1)),
            "min_frac": float(params.get("min_frac", 0.10)),
            "ann_factor": int(params.get("ann_factor", 252)),
            "gamma": float(params.get("gamma", 5.0)),
            "w_min": float(params.get("w_min", 0.0)),
            "w_max": float(params.get("w_max", 3.0)),
            "eps": float(params.get("eps", 1e-12)),
        }

    # ----------------------------
    # Strategy API
    # ----------------------------

    def fit(self, train: pd.DataFrame, spec: RunSpec) -> None:
        p = self.parse_params(spec)

        self.state_var_ = p["state_var"]
        self.lag_state_ = p["lag_state"]

        df = train.copy().sort_index()

        if self.state_var_ not in df.columns:
            raise ValueError(f"Train data missing state_var column '{self.state_var_}'")

        df["S_used"] = df[self.state_var_].shift(self.lag_state_)

        train_eff = df.dropna(subset=["S_used", p["return_col"]])
        if train_eff.empty or len(train_eff) < 20:
            self.tau_, self.w_low_, self.w_high_ = np.nan, 0.0, 0.0
            return

        tau = self.choose_tau_max_sep_sharpe(
            train_eff,
            state_var="S_used",
            return_col=p["return_col"],
            min_frac=p["min_frac"],
            ann_factor=p["ann_factor"],
        )

        if tau is None or np.isnan(tau):
            self.tau_, self.w_low_, self.w_high_ = np.nan, 0.0, 0.0
            return

        w_low, w_high = self.estimate_weights_meanvar(
            train_eff,
            state_var="S_used",
            return_col=p["return_col"],
            tau=tau,
            gamma=p["gamma"],
            w_min=p["w_min"],
            w_max=p["w_max"],
            eps=p["eps"],
        )

        self.tau_ = float(tau)
        self.w_low_ = float(w_low)
        self.w_high_ = float(w_high)

    def compute_weight(self, data: pd.DataFrame, spec: RunSpec) -> pd.Series:
        if self.tau_ is None:
            raise RuntimeError("compute_weight called before fit().")

        p = self.parse_params(spec)
        df = data.copy().sort_index()

        if p["state_var"] not in df.columns:
            raise ValueError(f"Data missing state_var column '{p['state_var']}'")

        S_used = df[p["state_var"]].shift(self.lag_state_)

        w = np.where(
            S_used.isna(),
            0.0,
            np.where(S_used > self.tau_, self.w_high_, self.w_low_)
        )

        return pd.Series(w, index=df.index, name="weight")

    # ----------------------------
    # Helpers
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
        """
        Choose tau to maximize |Sharpe_high - Sharpe_low|.
        Uses a quantile grid (robust, deterministic).
        """
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
        """
        Mean-variance weights per regime:
            w* = mu / (gamma * var)
        then clip to [w_min, w_max].
        """
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

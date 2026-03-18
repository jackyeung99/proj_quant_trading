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
    Outputs a binary signal / weight:
      signal[t] = 1 if S_used[t] > tau else 0

    Current implementation assumes a single-asset return series.
    The state variable may be:
      - asset-specific (from inputs.asset_features[state_var])
      - global/macro (from inputs.global_features[state_var])
    """

    def __init__(self) -> None:
        self.tau_: float | None = None
        self.w_low_: float | None = None
        self.w_high_: float | None = None
        self.state_var_: str | None = None
        self.state_source_: str | None = None  # "asset" or "global"

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
            "weight_type": str(params.get("weight_allocation", "binary")),
            "w_min": float(params.get("w_min", 0.0)),
            "w_max": float(params.get("w_high", 1.0)),
            "eps": float(params.get("eps", 1e-12)),
        }

    def get_model_state(self) -> dict:
        return {
            "tau_star": self.tau_,
            "w_low": self.w_low_,
            "w_high": self.w_high_,
            "state_var": self.state_var_,
            "state_source": self.state_source_,
        }

    def required_asset_features(self, spec: RunSpec) -> list[str]:
        p = self.parse_params(spec)
        state_var = p["state_var"]

        # If state_var is asset-specific, declare it here.
        # If it's macro/global, this strategy should instead request it
        # through required_global_features().
        #
        # For now, this method assumes asset features unless overridden
        # by a naming convention or a param flag.
        if spec.params and spec.params.get("state_source") == "global":
            return []
        return [state_var]

    def required_global_features(self, spec: RunSpec) -> list[str]:
        p = self.parse_params(spec)
        state_var = p["state_var"]

        if spec.params and spec.params.get("state_source") == "global":
            return [state_var]
        return []

    def _get_single_asset(self, inputs: ModelInputs) -> str:
        if not isinstance(inputs.ret, pd.DataFrame) or inputs.ret.shape[1] != 1:
            raise ValueError("StateSignalModel expects inputs.ret to be [T x 1].")
        return str(inputs.ret.columns[0])

    def _get_state_series(self, inputs: ModelInputs, spec: RunSpec) -> pd.Series:
        p = self.parse_params(spec)
        state_var = p["state_var"]
        asset = self._get_single_asset(inputs)

        # Prefer explicit source if provided
        explicit_source = (spec.params or {}).get("state_source")

        if explicit_source == "asset":
            if state_var not in inputs.asset_features:
                raise ValueError(
                    f"Requested asset state_var '{state_var}' not found in inputs.asset_features. "
                    f"Available asset features: {list(inputs.asset_features.keys())}"
                )
            panel = inputs.asset_features[state_var].sort_index()
            if asset not in panel.columns:
                raise ValueError(
                    f"Asset feature panel '{state_var}' missing asset '{asset}'. "
                    f"Available assets: {list(panel.columns)}"
                )
            self.state_source_ = "asset"
            return pd.to_numeric(panel[asset], errors="coerce")

        if explicit_source == "global":
            if state_var not in inputs.global_features.columns:
                raise ValueError(
                    f"Requested global state_var '{state_var}' not found in inputs.global_features. "
                    f"Available global features: {list(inputs.global_features.columns)}"
                )
            self.state_source_ = "global"
            return pd.to_numeric(inputs.global_features[state_var].sort_index(), errors="coerce")

        # Auto-detect if no explicit source provided
        if state_var in inputs.asset_features:
            panel = inputs.asset_features[state_var].sort_index()
            if asset not in panel.columns:
                raise ValueError(
                    f"Asset feature panel '{state_var}' missing asset '{asset}'. "
                    f"Available assets: {list(panel.columns)}"
                )
            self.state_source_ = "asset"
            return pd.to_numeric(panel[asset], errors="coerce")

        if state_var in inputs.global_features.columns:
            self.state_source_ = "global"
            return pd.to_numeric(inputs.global_features[state_var].sort_index(), errors="coerce")

        raise ValueError(
            f"state_var '{state_var}' not found in either asset_features or global_features.\n"
            f"asset_features keys: {list(inputs.asset_features.keys())}\n"
            f"global_features columns: {list(inputs.global_features.columns)}"
        )

    def get_persisted_series(self, *, test_inputs: ModelInputs, spec: RunSpec) -> pd.DataFrame:
        S = self._get_state_series(test_inputs, spec).reindex(test_inputs.ret.index)

        out = pd.DataFrame(
            {
                "state_value": S,
                "signal": (S > float(self.tau_)).astype(float) if self.tau_ is not None else np.nan,
                "tau_star": float(self.tau_) if self.tau_ is not None else np.nan,
                "w_low": self.w_low_,
                "w_high": self.w_high_,
            },
            index=test_inputs.ret.index,
        )
        return out

    def fit(self, inputs: ModelInputs, spec: RunSpec) -> None:
        p = self.parse_params(spec)
        self.state_var_ = p["state_var"]

        r = inputs.ret.sort_index()
        asset = self._get_single_asset(inputs)
        r1 = pd.to_numeric(r[asset], errors="coerce")

        S_used = self._get_state_series(inputs, spec).reindex(r.index)

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
            n_grid=p["n_grid"],
        )

        if tau is None or np.isnan(tau):
            self.tau_, self.w_low_, self.w_high_ = np.nan, 0.0, 0.0
            return

        if p["weight_type"] == "mean_var":
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
            w_low, w_high = 1.0, 0.0

        self.tau_ = float(tau)
        self.w_low_ = float(w_low)
        self.w_high_ = float(w_high)

    def predict(self, inputs: ModelInputs, spec: RunSpec) -> pd.DataFrame:
        if self.tau_ is None:
            raise RuntimeError("predict called before fit().")

        asset = self._get_single_asset(inputs)
        S_used = self._get_state_series(inputs, spec).reindex(inputs.ret.index)

        sig = (S_used > float(self.tau_)).astype(float)

        w_vals = np.where(
            sig.fillna(0.0).values > 0.5,
            self.w_high_,
            self.w_low_,
        )

        w = pd.DataFrame(
            w_vals,
            index=inputs.ret.index,
            columns=[asset],
            dtype=float,
        )

        return w.fillna(0.0)

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
        w_lo: float = 0.0,
        w_hi: float = 1.0,
        n_grid: int = 100,
        min_n: int = 5,
        eps: float = 1e-12,
        ddof: int = 1,
    ) -> tuple[float, float]:
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

            if (not np.isfinite(var)) or var < eps:
                return float(w_hi if mu > 0 else w_lo)

            U = grid * mu - 0.5 * float(gamma) * (grid ** 2) * var
            j = int(np.nanargmax(U))
            return float(grid[j])

        return best_w(low), best_w(high)
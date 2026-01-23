from __future__ import annotations
import pandas as pd
from qbt.core.types import RunSpec
from qbt.strategies.base import Strategy
from qbt.metrics.summary import _sharpe

import numpy as np
from scipy.optimize import minimize_scalar

class StateStrategy(Strategy):
    
    
    def compute_weight(self, data: pd.DataFrame, spec: RunSpec) -> pd.Series:
        # Always fully invested
        

        tau = self.optimize_tau(data)
        
        
        return


    def split(selff, df, state_var, tau):
        mask = df[state_var] <= tau
        return df[mask], df[~mask]

    def sharpe_diff(self, low, high, return_col):
        sharpe_low = _sharpe(low[return_col])
        sharpe_high= _sharpe(high[return_col])
        return sharpe_low, sharpe_high
        


    def ret_diff(self, low, high, return_col):
        return low[return_col].mean() - high[return_col].mean()

    def optimize_tau(self, df, state_var, return_col, min_frac=0.05):
        x = df[state_var].to_numpy()

        # Restrict tau to interior quantiles so both groups aren't tiny
        lo, hi = np.quantile(x, [min_frac, 1 - min_frac])

        def objective(tau):
            low, high = self.split(df, state_var, tau)
            # Penalize invalid splits
            if len(low) == 0 or len(high) == 0:
                return np.inf
            # maximize diff => minimize negative diff
            return -self.ret_diff(low, high, return_col)

        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        tau_star = res.x

        low_star, high_star = self.split(df, state_var, tau_star)
        best_diff = self.ret_diff(low_star, high_star, return_col)

        return {
            "tau": float(tau_star),
            "diff": float(best_diff),
            "n_low": int(len(low_star)),
            "n_high": int(len(high_star)),
            "result": res,
        }
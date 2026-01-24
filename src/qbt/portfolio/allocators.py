import pandas as pd 
import numpy as np 


from qbt.portfolio.portfolio_constructor import PortfolioConstructor

class HoldAllocator(PortfolioConstructor):
    def allocate(self, signals, spec, assets):
        idx = signals.index
        return pd.DataFrame(1.0, index=idx, columns=assets)



class TopKEqualWeightAllocator(PortfolioConstructor):
    def __init__(self, k: int):
        self.k = k

    def allocate(self, signals, spec, assets):
        if isinstance(signals, pd.Series):
            signals = signals.to_frame(name=assets[0])

        signals = signals.reindex(columns=assets)
        w = pd.DataFrame(0.0, index=signals.index, columns=assets)

        k = min(self.k, len(assets))
        topk = signals.rank(axis=1, ascending=False, method="first") <= k
        w[topk] = 1.0 / k
        return w


class MeanVarianceAllocator(PortfolioConstructor):
    """
    Mean-variance exposure sizing per asset:
        w_i = mu_i / (gamma * var_i)
    estimated on training returns, then held constant over the allocation window.

    - fit(...) estimates mu/var and stores weights per asset.
    - allocate(...) outputs a weights DataFrame aligned to signals.index.

    Notes:
    - This produces *exposure* weights, not necessarily fully-invested weights.
      You can combine with a normalizer if you want sum(abs(w))=1.
    """

    def __init__(
        self,
        gamma: float = 5.0,
        w_min: float = 0.0,
        w_max: float = 3.0,
        eps: float = 1e-12,
        min_obs: int = 20,
        use_ewm: bool = False,
        ewm_span: int = 60,
    ) -> None:
        self.gamma = float(gamma)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.eps = float(eps)
        self.min_obs = int(min_obs)
        self.use_ewm = bool(use_ewm)
        self.ewm_span = int(ewm_span)

        self._w_by_asset: dict[str, float] | None = None

    def fit(self, train_ret: pd.DataFrame, assets: list[str]) -> None:
        """
        train_ret: DataFrame [T x N] with columns=assets
        """
        r = train_ret.reindex(columns=assets).dropna(how="any")
        w_by_asset: dict[str, float] = {}

        for a in assets:
            x = r[a].dropna()
            w_by_asset[a] = self._mv_weight(x)

        self._w_by_asset = w_by_asset

    def allocate(self, signals, spec, assets):
        """
        Returns weights [T x N] for the allocation window.
        `signals` is only used for its index (and to stay consistent with your interface).
        """
        if self._w_by_asset is None:
            raise RuntimeError("MeanVarianceAllocator.allocate called before fit().")

        idx = signals.index
        w = pd.DataFrame(0.0, index=idx, columns=assets, dtype=float)

        for a in assets:
            w[a] = float(self._w_by_asset.get(a, 0.0))

        return w

    def _mv_weight(self, x: pd.Series) -> float:
        """
        Compute w = mu / (gamma * var), clipped.
        Optionally uses EWM mean/var.
        """
        x = x.dropna()
        if len(x) < self.min_obs:
            return float(np.clip(0.0, self.w_min, self.w_max))

        if self.use_ewm:
            mu = float(x.ewm(span=self.ewm_span, adjust=False).mean().iloc[-1])
            var = float(x.ewm(span=self.ewm_span, adjust=False).var(bias=False).iloc[-1])
        else:
            mu = float(x.mean())
            var = float(x.var(ddof=1))

        if not np.isfinite(var) or var < self.eps:
            return float(np.clip(0.0, self.w_min, self.w_max))

        w = mu / (self.gamma * var)
        return float(np.clip(w, self.w_min, self.w_max))
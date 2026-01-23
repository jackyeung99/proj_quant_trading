import numpy as np
import pandas as pd

import qbt.backtesting.engine as engine


class DummyBuyHold:
    """
    Minimal strategy stub.
    - fit: record calls
    - compute_weight: return 1.0 exposure for all rows passed in
    """
    def __init__(self):
        self.fit_calls = 0
        self.compute_calls = 0
        self.last_compute_index = None

    def fit(self, data: pd.DataFrame, spec) -> None:
        self.fit_calls += 1

    def compute_weight(self, data: pd.DataFrame, spec) -> pd.Series:
        self.compute_calls += 1
        self.last_compute_index = data.index
        return pd.Series(1.0, index=data.index, dtype=float)


def fake_simulator(data: pd.DataFrame, w, weight_lag: int):
    """
    A tiny deterministic simulator so tests don't depend on
    your simulator column naming / multiindex / etc.

    Expects:
      - data has column 'ret'
      - w is a Series indexed by date (or a DataFrame with one column)
    Returns columns that your engine expects: ret_net, equity_net, weight
    """
    if isinstance(w, pd.DataFrame):
        # assume single-asset weights DF
        ws = w.iloc[:, 0]
    else:
        ws = w

    ws = ws.reindex(data.index).fillna(0.0).astype(float)
    ws_lag = ws.shift(weight_lag).fillna(0.0)

    ret_net = ws_lag * data["ret"].astype(float)
    equity_net = (1.0 + ret_net).cumprod()

    out = pd.DataFrame(
        {
            "weight": ws_lag,
            "ret_net": ret_net,
            "equity_net": equity_net,
        },
        index=data.index,
    )
    return out


def fake_metrics(series: pd.Series) -> dict:
    # minimal metrics stub
    return {"mean": float(series.mean())}


def test_run_backtest_single_shot_smoke(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame({"ret": [0.01, -0.02, 0.03, 0.00]}, index=idx)

    # avoid disk I/O
    monkeypatch.setattr(engine, "load_data", lambda spec: df)

    # strategy map -> dummy strategy
    monkeypatch.setattr(engine, "_STRATEGY_MAP", {"BuyHold": DummyBuyHold})

    # deterministic simulator + metrics
    monkeypatch.setattr(engine, "simulate_strategy_execution", fake_simulator)
    monkeypatch.setattr(engine, "compute_metrics", fake_metrics)

    # minimal spec stub (attributes referenced by engine.run_backtest)
    class Spec:
        strategy_name = "BuyHold"
        universe = "TEST"
        data_path = "dummy.csv"
        weight_lag = 1
        params = {}
        tag = None
        # ret_col/date_col only used by real load_data, which we patched out
        ret_col = "ret"
        date_col = "date"

    res = engine.run_backtest(Spec())

    assert res.meta.strategy_name == "BuyHold"
    assert "ret_net" in res.timeseries.columns
    assert "equity_net" in res.timeseries.columns
    assert "weight" in res.timeseries.columns

    # lag means first day should have 0 exposure
    w = res.timeseries["weight"].to_numpy()
    assert np.allclose(w, np.array([0.0, 1.0, 1.0, 1.0]), atol=1e-12)

    # equity should start at 1.0 and be finite
    eq = res.timeseries["equity_net"].to_numpy()
    assert np.isfinite(eq).all()
    assert abs(eq[0] - 1.0) < 1e-12


def test_run_backtest_walk_forward_stitches_test_windows_only(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    df = pd.DataFrame({"ret": [0.01, 0.02, -0.01, 0.03, 0.00, 0.01]}, index=idx)

    # avoid disk I/O
    monkeypatch.setattr(engine, "load_data", lambda spec: df)

    # strategy map -> dummy strategy
    monkeypatch.setattr(engine, "_STRATEGY_MAP", {"BuyHold": DummyBuyHold})

    # deterministic simulator + metrics
    monkeypatch.setattr(engine, "simulate_strategy_execution", fake_simulator)
    monkeypatch.setattr(engine, "compute_metrics", fake_metrics)

    # stub WalkForwardSpec so engine.WalkForwardSpec(**wf_cfg) always works
    class DummyWF:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(engine, "WalkForwardSpec", DummyWF)

    # force a single WF split: train first 3 days, test last 3 days
    def fake_splits(index, wf):
        train_idx = index[:3]
        test_idx = index[3:]
        yield train_idx, test_idx

    monkeypatch.setattr(engine, "iter_walk_forward_splits", fake_splits)

    class Spec:
        strategy_name = "BuyHold"
        universe = "TEST"
        data_path = "dummy.csv"
        weight_lag = 1
        params = {"walk_forward": {"any": "thing"}}  # truthy => WF branch
        tag = None
        ret_col = "ret"
        date_col = "date"

    res = engine.run_backtest(Spec())

    ts = res.timeseries

    # in walk-forward we only set weights for the TEST window (last 3 days),
    # then the simulator lags by 1 day
    #
    # full_w: [0,0,0, 1,1,1]
    # lag=1 => weight series in output: [0,0,0, 0,1,1]
    expected_w = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    assert np.allclose(ts["weight"].to_numpy(), expected_w, atol=1e-12)

    # sanity: returns are weight * ret, first nonzero should be on day 5
    # (because test window starts day 4 but gets lagged to day 5)
    assert ts["ret_net"].iloc[:4].abs().sum() == 0.0
    assert np.isfinite(ts["equity_net"].to_numpy()).all()

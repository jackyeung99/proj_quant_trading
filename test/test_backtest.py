import numpy as np
import pandas as pd
import pytest

# Import your module under test
# Adjust this import to wherever your code lives
import qbt.backtesting.engine as engine


# -------------------------
# Strategy + adapter stubs
# -------------------------

class DummyStrategy:
    """
    Strategy stub compatible with your engine:
      - required_features(spec) -> list[str]
      - fit(ModelInputs, spec)
      - predict(ModelInputs, spec) -> Series or DataFrame
    """
    def __init__(self, mode="series", weight_value=1.0):
        self.mode = mode
        self.weight_value = float(weight_value)
        self.fit_calls = 0
        self.predict_calls = 0
        self.last_fit_index = None
        self.last_predict_index = None

    def required_features(self, spec):
        # Pretend the strategy needs two features
        return ["feat_a", "feat_b"]

    def fit(self, inputs, spec):
        self.fit_calls += 1
        self.last_fit_index = inputs.ret.index

    def predict(self, inputs, spec):
        self.predict_calls += 1
        self.last_predict_index = inputs.ret.index

        idx = inputs.ret.index
        cols = list(inputs.ret.columns)

        if self.mode == "series":
            # A single weight series -> engine should broadcast to all assets
            return pd.Series(self.weight_value, index=idx, name="w", dtype=float)

        if self.mode == "df":
            # Already [T x N]
            return pd.DataFrame(self.weight_value, index=idx, columns=cols, dtype=float)

        raise ValueError("Unknown mode")


class SpyDataAdapter:
    """
    DataAdapter stub compatible with your engine:
      - load(spec) -> raw df (could be anything)
      - prepare(raw, spec, required_cols=...) -> ModelInputs
    """
    def __init__(self, inputs):
        self._inputs = inputs
        self.load_calls = 0
        self.prepare_calls = 0
        self.last_required_cols = None
        self.last_raw = None

    def load(self, spec):
        self.load_calls += 1
        # raw can be anything; engine passes it into prepare()
        raw = pd.DataFrame({"raw": [1]})
        self.last_raw = raw
        return raw

    def prepare(self, raw, spec, required_cols):
        self.prepare_calls += 1
        self.last_required_cols = list(required_cols)
        return self._inputs


# -------------------------
# Helpers
# -------------------------

def make_inputs(idx, assets=("A", "B")):
    ret = pd.DataFrame(
        {a: np.linspace(0.01, 0.01, len(idx)) for a in assets},
        index=idx,
        dtype=float,
    )
    features = pd.DataFrame(
        {
            "feat_a": np.arange(len(idx), dtype=float),
            "feat_b": np.arange(len(idx), dtype=float) + 100.0,
        },
        index=idx,
    )
    return engine.ModelInputs(ret=ret, features=features)


def fake_simulator(ret_df, w_df, weight_lag, transaction_cost_bps, **kwargs):
    """
    Deterministic simulator stub:
    - returns a dataframe containing 'port_ret_gross' so compute_metrics() works
    - checks alignment and uses lagged weights
    """
    assert isinstance(ret_df, pd.DataFrame)
    assert isinstance(w_df, pd.DataFrame)
    assert ret_df.index.equals(w_df.index)
    assert list(ret_df.columns) == list(w_df.columns)

    w_lag = w_df.shift(weight_lag).fillna(0.0)
    # Simple portfolio return: equal-weighted across assets after applying weights
    port_ret = (w_lag * ret_df).mean(axis=1)

    out = pd.DataFrame({"port_ret_gross": port_ret}, index=ret_df.index)
    return out


def fake_metrics(s):
    return {"mean": float(pd.Series(s).mean())}


class Spec:
    strategy_name = "Dummy"
    universe = "TEST"
    data = {"path": "dummy.parquet"}
    params = {"x": 1}
    tag = "taggy"


class BT:
    def __init__(self, weight_lag=1, transaction_cost_bps=0.0):
        self.use_walk_forward = False
        self.weight_lag = weight_lag
        self.transaction_cost_bps = transaction_cost_bps
        self.train_size = .8 
        self.test_size = .2
        self.min_train = 1
        self.expanding = False


# -------------------------
# Tests: _normalize_weights_to_df
# -------------------------

def test_normalize_weights_series_broadcasts_to_assets():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    assets = ["A", "B", "C"]
    w = pd.Series([1.0, 0.5, 0.0], index=idx)

    out = engine._normalize_weights_to_df(w, index=idx, assets=assets)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == assets
    assert out.shape == (3, 3)
    assert np.allclose(out["A"].to_numpy(), [1.0, 0.5, 0.0])
    assert np.allclose(out["B"].to_numpy(), [1.0, 0.5, 0.0])
    assert np.allclose(out["C"].to_numpy(), [1.0, 0.5, 0.0])


def test_normalize_weights_df_reindexes_and_fills_missing():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    assets = ["A", "B"]
    # Missing last date + missing column B
    w_df = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx[:3])

    out = engine._normalize_weights_to_df(w_df, index=idx, assets=assets)
    assert out.shape == (4, 2)
    # Column B should be filled with 0.0
    assert np.allclose(out["B"].to_numpy(), 0.0)
    # Last date should be filled with 0.0
    assert out.loc[idx[-1], "A"] == 0.0


# -------------------------
# Tests: build_weights (no WF vs WF)
# -------------------------

def test_build_weights_no_walk_forward_calls_fit_once_and_predict_once(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    inputs = make_inputs(idx, assets=("A", "B"))
    strat = DummyStrategy(mode="series", weight_value=1.0)

    # bt=None triggers the "no walk-forward" branch in your build_weights()
    w = engine.build_weights(
        inputs=inputs,
        strat=strat,
        spec=Spec(),
        assets=["A", "B"],
        bt=None,
    )

    assert strat.fit_calls == 1
    assert strat.predict_calls == 1
    assert w.shape == (5, 2)
    assert np.allclose(w.to_numpy(), 1.0)


def test_build_weights_walk_forward_stitches_only_test_windows(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    inputs = make_inputs(idx, assets=("A", "B"))
    strat = DummyStrategy(mode="series", weight_value=1.0)

    # Force 2 splits:
    # train: [0,1,2] test: [3,4]
    # train: [0,1,2,3,4] test: [5]
    def fake_splits(index, bt):
        yield index[:3], index[3:5]
        yield index[:5], index[5:]

    monkeypatch.setattr(engine, "iter_walk_forward_splits", fake_splits)

    bt = BT(weight_lag=1, transaction_cost_bps=0.0)

    w = engine.build_weights(
        inputs=inputs,
        strat=strat,
        spec=Spec(),
        assets=["A", "B"],
        bt=bt,
    )

    # full_w starts at zeros and only fills test windows
    # dates 0,1,2 never in test => should be 0
    assert np.allclose(w.loc[idx[:3]].to_numpy(), 0.0)
    # dates 3,4,5 are in test windows => should be 1
    assert np.allclose(w.loc[idx[3:]].to_numpy(), 1.0)

    # fit/predict called once per split
    assert strat.fit_calls == 2
    assert strat.predict_calls == 2


# -------------------------
# Tests: BacktestEngine.run integration (plumbing)
# -------------------------

def test_engine_run_plumbs_required_features_and_calls_simulator(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    inputs = make_inputs(idx, assets=("A", "B"))
    adapter = SpyDataAdapter(inputs=inputs)

    # Stub create_strategy -> returns our DummyStrategy
    dummy = DummyStrategy(mode="series", weight_value=1.0)
    monkeypatch.setattr(engine, "create_strategy", lambda name: dummy)

    # Stub simulator + metrics
    monkeypatch.setattr(engine, "simulate_strategy_execution", fake_simulator)
    monkeypatch.setattr(engine, "compute_metrics", fake_metrics)

    # Make run_id deterministic for assertions
    monkeypatch.setattr(engine, "make_run_id", lambda s, u: "RUN123")

    eng = engine.BacktestEngine(data_adapter=adapter)
    bt = BT(weight_lag=2, transaction_cost_bps=7.5)

    res = eng.run(Spec(), bt)

    # Adapter usage
    assert adapter.load_calls == 1
    assert adapter.prepare_calls == 1
    assert adapter.last_required_cols == ["feat_a", "feat_b"]

    # Strategy usage
    assert dummy.fit_calls >= 1
    assert dummy.predict_calls >= 1

    # Timeseries must include what engine uses for metrics
    assert "port_ret_gross" in res.timeseries.columns
    assert res.meta.run_id == "RUN123"
    assert res.meta.strategy_name == "Dummy"
    assert res.meta.universe == "TEST"
    assert res.meta.weight_lag == 2
    assert res.meta.params == {"x": 1}
    assert res.meta.tag == "taggy"

    # Metrics stubbed
    assert "mean" in res.metrics
    assert np.isfinite(res.metrics["mean"])


def test_engine_run_multi_asset_predict_df_not_broadcast(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    inputs = make_inputs(idx, assets=("A", "B", "C"))
    adapter = SpyDataAdapter(inputs=inputs)

    # Strategy returns a DF with per-asset weights (not broadcast)
    dummy = DummyStrategy(mode="df", weight_value=0.25)
    monkeypatch.setattr(engine, "create_strategy", lambda name: dummy)

    # Intercept simulator to assert weights are passed as [T x N] with correct columns
    def sim_assert(ret_df, w_df, weight_lag, transaction_cost_bps, **kwargs):
        assert list(w_df.columns) == ["A", "B", "C"]
        assert w_df.shape == (4, 3)
        assert np.allclose(w_df.to_numpy(), 0.25)
        return pd.DataFrame({"port_ret_gross": np.zeros(len(ret_df))}, index=ret_df.index)

    monkeypatch.setattr(engine, "simulate_strategy_execution", sim_assert)
    monkeypatch.setattr(engine, "compute_metrics", fake_metrics)
    monkeypatch.setattr(engine, "make_run_id", lambda s, u: "RUN456")

    eng = engine.BacktestEngine(data_adapter=adapter)
    bt = BT(weight_lag=1, transaction_cost_bps=0.0)
    res = eng.run(Spec(), bt)

    assert res.meta.run_id == "RUN456"
    assert "port_ret_gross" in res.timeseries.columns


def test_engine_run_passes_transaction_cost_bps(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    inputs = make_inputs(idx, assets=("A", "B"))
    adapter = SpyDataAdapter(inputs=inputs)

    dummy = DummyStrategy(mode="series", weight_value=1.0)
    monkeypatch.setattr(engine, "create_strategy", lambda name: dummy)

    seen = {}
    def sim_capture(ret_df, w_df, weight_lag, transaction_cost_bps, **kwargs):
        seen["weight_lag"] = weight_lag
        seen["transaction_cost_bps"] = transaction_cost_bps
        return pd.DataFrame({"port_ret_gross": np.zeros(len(ret_df))}, index=ret_df.index)

    monkeypatch.setattr(engine, "simulate_strategy_execution", sim_capture)
    monkeypatch.setattr(engine, "compute_metrics", fake_metrics)
    monkeypatch.setattr(engine, "make_run_id", lambda s, u: "RUN789")

    eng = engine.BacktestEngine(data_adapter=adapter)
    bt = BT(weight_lag=3, transaction_cost_bps=12.3)
    _ = eng.run(Spec(), bt)

    assert seen["weight_lag"] == 3
    assert seen["transaction_cost_bps"] == 12.3

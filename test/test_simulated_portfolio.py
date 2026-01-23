import numpy as np
import pandas as pd

from qbt.execution.simulator import simulate_strategy_execution


def test_compute_performance_applies_lag_no_lookahead():
    # returns: +10%, -10%, +10%
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    returns = pd.DataFrame([0.10, -0.10, 0.10], index=idx)
    # make the single asset name explicit (avoids weight_0)
    returns.columns = ["AAA"]

    # weights decided at time t, applied at t+1 when weight_lag=1
    weights = pd.DataFrame([1.0, 1.0, 1.0], index=idx)
    weights.columns = ["AAA"]

    ts = simulate_strategy_execution(returns=returns, weights=weights, weight_lag=1)

    # day1 position should be 0 (shifted), day2/3 should be 1
    assert ts["weight_AAA"].tolist() == [0.0, 1.0, 1.0]

    # realized portfolio returns = position * return (no costs)
    expected_ret = [0.0, -0.10, 0.10]
    assert np.allclose(ts["port_ret_net"].to_numpy(), np.array(expected_ret), atol=1e-12)

    # equity starts at 1.0 then compounds
    assert np.allclose(ts["equity_net"].to_numpy(), np.array([1.0, 0.9, 0.99]), atol=1e-12)


def test_compute_performance_aligns_and_fills_missing_weights_with_zero():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    returns = pd.DataFrame([0.01, 0.02, 0.03, 0.04], index=idx)
    returns.columns = ["AAA"]

    # provide weights only for last 2 days (like walk-forward stitching)
    weights = pd.DataFrame([1.0, 1.0], index=idx[-2:])
    weights.columns = ["AAA"]

    ts = simulate_strategy_execution(returns=returns, weights=weights, weight_lag=0)

    # missing weights should be treated as 0 exposure
    assert ts["weight_AAA"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert np.allclose(ts["port_ret_net"].to_numpy(), np.array([0.0, 0.0, 0.03, 0.04]), atol=1e-12)


def test_compute_performance_handles_all_zero_weights():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    returns = pd.DataFrame([0.05, -0.02, 0.01], index=idx)
    returns.columns = ["AAA"]

    weights = pd.DataFrame([0.0, 0.0, 0.0], index=idx)
    weights.columns = ["AAA"]

    ts = simulate_strategy_execution(returns=returns, weights=weights, weight_lag=1)

    assert np.allclose(ts["port_ret_net"].to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(ts["equity_net"].to_numpy(), 1.0, atol=1e-12)

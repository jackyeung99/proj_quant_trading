

import numpy as np
import pandas as pd
import pytest

from qbt.features.feature_engineering import aggregate_intra_bars as m



def _fake_compute_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """
    Deterministic returns for testing.
    - log: log(p_t) - log(p_{t-1})
    - simple: p_t/p_{t-1} - 1
    """
    if kind == "log":
        return np.log(prices).diff()
    if kind == "simple":
        return prices.pct_change()
    raise ValueError(kind)


def test_aggregate_intra_bars_cutoff_hour_splits_at_1500(monkeypatch):
    """
    With cutoff_hour=15.0 and 15m close-stamped bars:
      - A bar at 15:00 belongs to the period that ENDS at 15:00.
      - A bar at 15:15 belongs to the NEXT period.
    """
    monkeypatch.setattr(m, "compute_returns", _fake_compute_returns)

    tz = "America/New_York"
    idx = pd.DatetimeIndex(
        [
            "2020-01-01 14:45",
            "2020-01-01 15:00",  # boundary bar
            "2020-01-01 15:15",  # should roll into next period
            "2020-01-02 15:00",  # next boundary
        ],
        tz=tz,
    )

    # One asset with prices chosen so returns are easy:
    # Period 1: 14:45 -> 15:00
    # Period 2: 15:15 -> next day 15:00
    prices = pd.DataFrame({"XLE": [100.0, 110.0, 200.0, 220.0]}, index=idx)

    out = m.aggregate_intra_bars(
        prices,
        freq="1D",
        cutoff_hour=15.0,
        return_kind="log",
        tz=None,  # already tz-aware
    )

    # Expect exactly 2 daily bins
    assert out.index.size == 2

    # Pull the XLE slice
    x = out["XLE"]

    # --- Period 1 expectations ---
    # It should include 14:45 and 15:00 (2 price bars)
    # Returns are diff(log(price)), so within period:
    #   ret_sum = log(110) - log(100) = log(110/100)
    #   rvar_sum = (log(110/100))^2
    #   n_intra counts non-NaN returns => 1 return for 2 price points
    p1_open = 100.0
    p1_close = 110.0
    p1_ret = float(np.log(p1_close / p1_open))
    p1_rvar = p1_ret ** 2
    p1_rvol = float(np.sqrt(p1_rvar))
    p1_n_intra = 1

    assert np.isclose(x["open"].iloc[0], p1_open)
    assert np.isclose(x["close"].iloc[0], p1_close)
    assert np.isclose(x["ret"].iloc[0], p1_ret)
    assert np.isclose(x["rvar"].iloc[0], p1_rvar)
    assert np.isclose(x["rvol"].iloc[0], p1_rvol)
    assert int(x["n_intra"].iloc[0]) == p1_n_intra

    # --- Period 2 expectations ---
    # It should include 15:15 (Jan 1) and 15:00 (Jan 2)
    p2_open = 200.0
    p2_close = 220.0
    p2_ret = float(np.log(p2_close / p2_open))
    p2_rvar = p2_ret ** 2
    p2_rvol = float(np.sqrt(p2_rvar))
    p2_n_intra = 1

    assert np.isclose(x["open"].iloc[1], p2_open)
    assert np.isclose(x["close"].iloc[1], p2_close)
    assert np.isclose(x["ret"].iloc[1], p2_ret)
    assert np.isclose(x["rvar"].iloc[1], p2_rvar)
    assert np.isclose(x["rvol"].iloc[1], p2_rvol)
    assert int(x["n_intra"].iloc[1]) == p2_n_intra


def test_aggregate_intra_bars_localizes_naive_index_when_tz_provided(monkeypatch):
    monkeypatch.setattr(m, "compute_returns", _fake_compute_returns)

    # Naive timestamps (no tz)
    idx = pd.date_range("2020-01-01 14:45", periods=4, freq="15min")
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0]}, index=idx)

    out = m.aggregate_intra_bars(
        prices,
        freq="1D",
        cutoff_hour=15.0,
        return_kind="log",
        tz="America/New_York",
    )

    # Output should be tz-aware now
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is not None
    assert str(out.index.tz) in ("America/New_York", "US/Eastern")  # pandas aliasing can vary


def test_aggregate_intra_bars_raises_if_naive_index_and_no_tz(monkeypatch):
    monkeypatch.setattr(m, "compute_returns", _fake_compute_returns)

    idx = pd.date_range("2020-01-01 09:30", periods=3, freq="15min")
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=idx)

    with pytest.raises(ValueError, match="tz-aware|pass tz"):
        m.aggregate_intra_bars(prices, freq="1D", cutoff_hour=15.0, tz=None)

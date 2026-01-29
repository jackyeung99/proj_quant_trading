import pandas as pd

from qbt.backtesting.splitter import iter_walk_forward_splits
from qbt.core.types import BacktestSpec


def test_iter_walk_forward_splits_rolling_windows():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")

    wf = BacktestSpec(train_size=4, test_size=2, expanding=False, min_train=1)

    splits = list(iter_walk_forward_splits(idx, wf))

    # expected:
    # train [0:4), test [4:6)
    # train [2:6), test [6:8)
    # train [4:8), test [8:10)
    assert len(splits) == 3

    tr0, te0 = splits[0]
    assert list(tr0) == list(idx[0:4])
    assert list(te0) == list(idx[4:6])

    tr1, te1 = splits[1]
    assert list(tr1) == list(idx[2:6])
    assert list(te1) == list(idx[6:8])

    tr2, te2 = splits[2]
    assert list(tr2) == list(idx[4:8])
    assert list(te2) == list(idx[8:10])

def test_iter_walk_forward_splits_expanding_windows():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    wf = BacktestSpec(train_size=4, test_size=2, expanding=True, min_train=1)

    splits = list(iter_walk_forward_splits(idx, wf))

    # expected:
    # train [0:4), test [4:6)
    # train [0:6), test [6:8)
    # train [0:8), test [8:10)
    assert len(splits) == 3

    tr0, te0 = splits[0]
    assert list(tr0) == list(idx[0:4])
    assert list(te0) == list(idx[4:6])

    tr1, te1 = splits[1]
    assert list(tr1) == list(idx[0:6])
    assert list(te1) == list(idx[6:8])

    tr2, te2 = splits[2]
    assert list(tr2) == list(idx[0:8])
    assert list(te2) == list(idx[8:10])

def test_iter_walk_forward_splits_insufficient_data():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    wf = BacktestSpec(train_size=4, test_size=2, expanding=False)

    splits = list(iter_walk_forward_splits(idx, wf))

    # 4 train + 2 test > 5 total → no valid split
    assert splits == []
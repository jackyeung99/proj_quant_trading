from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class WalkForwardSpec:
    train_size: int          # number of rows/days in train window
    test_size: int           # number of rows/days in test window
    step_size: int | None = None   # default = test_size
    expanding: bool = False        # if True, train expands instead of rolling
    min_train: int | None = None   # optional warmup guard

def iter_walk_forward_splits(index: pd.Index, wf: WalkForwardSpec):
    n = len(index)
    step = wf.step_size or wf.test_size
    min_train = wf.min_train or wf.train_size

    start = 0
    while True:
        if wf.expanding:
            train_start = 0
            train_end = start + wf.train_size
        else:
            train_start = start
            train_end = start + wf.train_size

        test_start = train_end
        test_end = test_start + wf.test_size

        if test_end > n:
            break

        if (train_end - train_start) >= min_train:
            train_idx = index[train_start:train_end]
            test_idx = index[test_start:test_end]
            yield train_idx, test_idx

        start += step

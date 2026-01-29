from __future__ import annotations


import math
from typing import Iterator, Tuple, Union
import pandas as pd

from qbt.core.types import SizeLike, BacktestSpec


def _parse_size(value: SizeLike, n: int, *, name: str, allow_none: bool = True) -> int | None:
    """
    Convert SizeLike into an integer number of rows.
    - int -> rows
    - float in (0,1] -> fraction of n
    - str: "60%" -> percent of n, "0.6" -> fraction, "252" -> rows
    Returns int (>=1) or None if value is None and allow_none.
    """
    if value is None:
        return None if allow_none else 0

    # int: rows
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
        return value

    # float: fraction
    if isinstance(value, float):
        if not (0.0 < value <= 1.0):
            raise ValueError(f"{name} float must be in (0, 1], got {value}")
        return max(1, int(math.ceil(value * n)))

    # str: percent/fraction/rows
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError(f"{name} cannot be empty string")

        if s.endswith("%"):
            pct = float(s[:-1].strip())
            if not (0.0 < pct <= 100.0):
                raise ValueError(f"{name} percent must be in (0, 100], got {pct}%")
            frac = pct / 100.0
            return max(1, int(math.ceil(frac * n)))

        # try numeric string
        num = float(s)
        if num.is_integer() and num >= 1:
            return int(num)

        # otherwise treat as fraction
        if not (0.0 < num <= 1.0):
            raise ValueError(f"{name} string numeric must be rows (>=1) or fraction in (0,1], got {value!r}")
        return max(1, int(math.ceil(num * n)))

    raise TypeError(f"{name} must be int, float, str, or None; got {type(value).__name__}")


def iter_walk_forward_splits(
    index: pd.Index,
    bt: BacktestSpec,
) -> Iterator[Tuple[pd.Index, pd.Index]]:
    """
    Yield (train_index, test_index) splits over a time index.

    Conventions (position-based slicing, end-exclusive):
      - rolling (expanding=False):
          train window length = train_size
          step size = test_size
      - expanding (expanding=True):
          first train window length = train_size
          then grows by test_size each split
          step size = test_size

    Stops when a full test window of length test_size cannot be formed.
    """
    n = len(index)
    if n == 0:
        return

    train = _parse_size(bt.train_size, n, name="train_size", allow_none=False)
    test = _parse_size(bt.test_size, n, name="test_size", allow_none=False)
    min_train = _parse_size(bt.min_train, n, name="min_train", allow_none=True)

    if min_train is None:
        min_train = train

    if train <= 0 or test <= 0:
        return

    # We advance by test window size (standard walk-forward CV)
    step = getattr(bt, "step_size", None)
    if step is None:
        step = test
    step = _parse_size(step, n, name="step_size", allow_none=False)

    # Start with a training end at least min_train (and typically train)
    train_end = max(train, min_train)

    # Safety: must be able to fit at least one test window
    if train_end + test > n:
        return

    while True:
        test_start = train_end
        test_end = test_start + test

        if test_end > n:
            break

        if bt.expanding:
            train_start = 0
        else:
            train_start = max(0, train_end - train)

        
        yield index[train_start:train_end], index[test_start:test_end]

        train_end += step

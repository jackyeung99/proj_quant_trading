from __future__ import annotations


import math
from typing import Iterator, Tuple, Union
import pandas as pd

from qbt.core.types import SizeLike, WalkForwardSpec


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


def iter_walk_forward_splits(index: pd.Index, wf: WalkForwardSpec) -> Iterator[Tuple[pd.Index, pd.Index]]:
    n = len(index)
    if n == 0:
        return

    train = _parse_size(wf.train_size, n, name="train_size", allow_none=False)
    test = _parse_size(wf.test_size, n, name="test_size", allow_none=False)
    step = _parse_size(wf.step_size, n, name="step_size", allow_none=True)
    min_train = _parse_size(wf.min_train, n, name="min_train", allow_none=True)

    # Defaults after parsing
    if step is None:
        step = test
    if min_train is None:
        min_train = train

    if train + test > n:
        # Not enough data for even one split
        return

    start = 0
    while True:
        if wf.expanding:
            train_start = 0
            train_end = start + train
        else:
            train_start = start
            train_end = start + train

        test_start = train_end
        test_end = test_start + test

        if test_end > n:
            break

        if (train_end - train_start) >= min_train:
            yield index[train_start:train_end], index[test_start:test_end]

        start += step

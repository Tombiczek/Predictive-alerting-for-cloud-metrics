from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import make_sliding_windows


def test_make_sliding_windows_builds_expected_labels() -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="5min")
    dataset = pd.DataFrame(
        {
            "series_id": ["s1"] * 10,
            "timestamp": timestamps,
            "value": np.arange(10, dtype=float),
            "is_incident": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        }
    )

    x, y = make_sliding_windows(dataset=dataset, window_size=3, horizon=2)

    assert x.shape == (6, 3, 1)
    assert y.shape == (6,)
    assert y.tolist() == [1, 1, 0, 1, 1, 0]



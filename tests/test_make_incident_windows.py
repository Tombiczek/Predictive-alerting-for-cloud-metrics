from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.windowing import make_sliding_windows


def test_make_sliding_windows_uses_incident_starts_and_keeps_clean_history_only() -> None:
    timestamps = pd.date_range("2024-01-01", periods=9, freq="5min")
    dataset = pd.DataFrame(
        {
            "series_id": ["s1"] * 9,
            "timestamp": timestamps,
            "value": np.arange(9, dtype=float),
            "is_incident": [0, 0, 0, 1, 1, 0, 0, 0, 0],
        }
    )

    x, y, meta = make_sliding_windows(dataset=dataset, window_size=2, horizon=2)

    assert x.shape == (3, 1, 2)
    assert y.tolist() == [1, 1, 0]
    np.testing.assert_array_equal(x[:, 0, :], np.array([[0.0, 1.0], [1.0, 2.0], [5.0, 6.0]]))

    assert meta["series_id"].tolist() == ["s1", "s1", "s1"]
    assert meta["t_end_idx"].tolist() == [1, 2, 6]
    assert meta["t_end_ts"].tolist() == [timestamps[1], timestamps[2], timestamps[6]]


def test_make_sliding_windows_returns_empty_structures_when_no_valid_samples() -> None:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="5min")
    dataset = pd.DataFrame(
        {
            "series_id": ["s1"] * 5,
            "timestamp": timestamps,
            "value": np.arange(5, dtype=float),
            "is_incident": [1, 1, 1, 1, 1],
        }
    )

    x, y, meta = make_sliding_windows(dataset=dataset, window_size=2, horizon=1)

    assert x.shape == (0, 1, 2)
    assert y.dtype == np.int8
    assert y.size == 0
    assert meta.empty

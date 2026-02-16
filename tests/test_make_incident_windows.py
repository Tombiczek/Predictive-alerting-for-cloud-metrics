from __future__ import annotations

import numpy as np
import pandas as pd

from predictive_alerting.dataset_builder import make_incident_windows


def test_make_incident_windows_builds_expected_labels() -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="5min")
    dataset = pd.DataFrame(
        {
            "series_id": ["s1"] * 10,
            "timestamp": timestamps,
            "value": np.arange(10, dtype=float),
            "is_incident": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        }
    )

    x, y, meta = make_incident_windows(dataset=dataset, window_size=3, horizon=2)

    assert x.shape == (6, 3, 1)
    assert y.shape == (6,)
    assert y.tolist() == [1, 1, 0, 1, 1, 0]
    assert len(meta) == 6
    assert meta.iloc[0]["window_start"] == timestamps[0]
    assert meta.iloc[0]["horizon_start"] == timestamps[3]


def test_make_incident_windows_skips_nan_windows_when_dropna_true() -> None:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="5min")
    dataset = pd.DataFrame(
        {
            "series_id": ["s1"] * 6,
            "timestamp": timestamps,
            "value": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
            "is_incident": [0, 0, 0, 0, 1, 0],
        }
    )

    x, y, _ = make_incident_windows(dataset=dataset, window_size=2, horizon=2, dropna=True)

    assert x.shape == (1, 2, 1)
    assert y.tolist() == [0]

import pandas as pd
import pytest

from src.data import normalize_series


def test_normalize_series_applies_zscore_per_series() -> None:
    dataset = pd.DataFrame({
        "series_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
        "timestamp": pd.date_range("2024-01-01", periods=6, freq="5min"),
        "value": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        "is_incident": [0, 0, 0, 0, 0, 0],
    })

    result = normalize_series(dataset)

    # Each series should have mean ~0 and std ~1
    s1_values = result[result["series_id"] == "s1"]["value"]
    s2_values = result[result["series_id"] == "s2"]["value"]

    assert s1_values.mean() == pytest.approx(0.0, abs=1e-10)
    assert s2_values.mean() == pytest.approx(0.0, abs=1e-10)
    assert s1_values.std() == pytest.approx(1.0, abs=0.1)
    assert s2_values.std() == pytest.approx(1.0, abs=0.1)


def test_normalize_series_does_not_modify_original() -> None:
    dataset = pd.DataFrame({
        "series_id": ["s1", "s1"],
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="5min"),
        "value": [10.0, 20.0],
        "is_incident": [0, 0],
    })
    original_values = dataset["value"].tolist()

    normalize_series(dataset)

    assert dataset["value"].tolist() == original_values


def test_normalize_series_handles_zero_std() -> None:
    dataset = pd.DataFrame({
        "series_id": ["s1", "s1", "s1"],
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min"),
        "value": [5.0, 5.0, 5.0],  # All same values -> std = 0
        "is_incident": [0, 0, 0],
    })

    result = normalize_series(dataset)

    # Should return 0.0 when std is 0
    assert result["value"].tolist() == [0.0, 0.0, 0.0]



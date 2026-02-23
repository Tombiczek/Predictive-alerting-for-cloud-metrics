import numpy as np
import pandas as pd
import pytest

from src.data.features import extract_features
from src.data.loading import normalize_series


def test_normalize_series_applies_zscore_per_series() -> None:
    dataset = pd.DataFrame(
        {
            "series_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="5min"),
            "value": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "is_incident": [0, 0, 0, 0, 0, 0],
        }
    )

    result = normalize_series(dataset)

    s1_values = result[result["series_id"] == "s1"]["value"]
    s2_values = result[result["series_id"] == "s2"]["value"]

    assert s1_values.mean() == pytest.approx(0.0, abs=1e-12)
    assert s2_values.mean() == pytest.approx(0.0, abs=1e-12)
    assert s1_values.std() == pytest.approx(1.0, abs=1e-12)
    assert s2_values.std() == pytest.approx(1.0, abs=1e-12)


def test_normalize_series_does_not_modify_original() -> None:
    dataset = pd.DataFrame(
        {
            "series_id": ["s1", "s1"],
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="5min"),
            "value": [10.0, 20.0],
            "is_incident": [0, 0],
        }
    )
    original_values = dataset["value"].tolist()

    normalize_series(dataset)

    assert dataset["value"].tolist() == original_values


def test_normalize_series_handles_zero_std() -> None:
    dataset = pd.DataFrame(
        {
            "series_id": ["s1", "s1", "s1"],
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "value": [5.0, 5.0, 5.0],
            "is_incident": [0, 0, 0],
        }
    )

    result = normalize_series(dataset)

    assert result["value"].tolist() == [0.0, 0.0, 0.0]


def test_extract_features_builds_expected_feature_vector() -> None:
    x_windows = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[2.0, 2.0, 2.0, 2.0]],
        ]
    )

    result = extract_features(x_windows)

    assert result.shape == (2, 10)
    expected_first = np.array([2.5, 1.11803399, 1.0, 4.0, 3.0, 3.7, 1.3, 1.0, 2.5, 1.5])
    expected_second = np.array([2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(result[0], expected_first, atol=1e-7)
    np.testing.assert_allclose(result[1], expected_second, atol=1e-7)

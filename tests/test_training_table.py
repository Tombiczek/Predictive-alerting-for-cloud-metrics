from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictive_alerting.dataset_builder import (
    build_training_table,
    build_window_features,
    save_training_dataset_parquet,
)


def test_build_window_features_computes_expected_stats() -> None:
    windows_x = np.array(
        [
            [[0.0], [1.0], [2.0]],
            [[2.0], [2.0], [2.0]],
        ]
    )

    features = build_window_features(windows_x, feature_columns=("value",))

    assert list(features.columns) == [
        "value_mean",
        "value_std",
        "value_min",
        "value_max",
        "value_last",
        "value_first",
        "value_diff_last_first",
        "value_slope",
        "value_median",
        "value_q25",
        "value_q75",
    ]
    assert features.shape == (2, 11)

    first_row = features.iloc[0]
    assert first_row["value_mean"] == pytest.approx(1.0)
    assert first_row["value_std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert first_row["value_min"] == pytest.approx(0.0)
    assert first_row["value_max"] == pytest.approx(2.0)
    assert first_row["value_last"] == pytest.approx(2.0)
    assert first_row["value_first"] == pytest.approx(0.0)
    assert first_row["value_diff_last_first"] == pytest.approx(2.0)
    assert first_row["value_slope"] == pytest.approx(1.0)


def test_build_training_table_creates_features_and_label_without_leakage() -> None:
    timestamps = pd.date_range("2024-01-01", periods=7, freq="5min")
    dataset = pd.DataFrame(
        {
            "series_id": ["s1"] * 7,
            "timestamp": timestamps,
            "value": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0],
            "is_incident": [0, 0, 0, 0, 1, 0, 0],
        }
    )

    training_df = build_training_table(dataset=dataset, window_size=3, horizon=2)

    # 7 points -> 3 windows for W=3 and H=2
    assert len(training_df) == 3

    # First row features come from [1,2,3], label comes from next two points [0,1]
    assert training_df.iloc[0]["value_mean"] == pytest.approx(2.0)
    assert training_df.iloc[0]["value_last"] == pytest.approx(3.0)
    assert int(training_df.iloc[0]["label"]) == 1

    # Second row label comes from [1,0] in the next horizon
    assert int(training_df.iloc[1]["label"]) == 1

    # Third row label comes from [0,0] in the next horizon
    assert int(training_df.iloc[2]["label"]) == 0


def test_save_training_dataset_parquet_creates_parent_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved_paths: list[Path] = []

    def fake_to_parquet(self: pd.DataFrame, path: Path, **kwargs: object) -> None:
        saved_paths.append(Path(path))
        Path(path).write_text("parquet-bytes-placeholder")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    training_df = pd.DataFrame({"label": [0, 1]})
    output_path = tmp_path / "test_artifacts" / "training.parquet"

    saved = save_training_dataset_parquet(training_df, output_path)

    assert saved == output_path
    assert saved.exists()
    assert saved_paths == [output_path]


def test_save_training_dataset_parquet_raises_helpful_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_to_parquet(self: pd.DataFrame, path: Path, **kwargs: object) -> None:
        raise ImportError("no engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    training_df = pd.DataFrame({"label": [0, 1]})

    with pytest.raises(ImportError, match="pyarrow|fastparquet"):
        save_training_dataset_parquet(training_df, tmp_path / "test_artifacts" / "fail.parquet")

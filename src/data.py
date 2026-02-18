import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_incident_windows(labels_path: Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Load incident time windows from NAB's combined_windows.json format."""
    raw = json.loads(labels_path.read_text())

    result = {}
    for series_id, intervals in raw.items():
        windows = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in intervals]
        result[series_id] = windows

    return result


def build_labeled_dataset(
    series_paths: list[Path],
    labels_path: Path,
) -> pd.DataFrame:
    """
    Load multiple CSV files and label each row with incident info.

    Returns a DataFrame with columns:
    - series_id: identifier derived from filename
    - timestamp, value: the raw metric data
    - is_incident: 1 if this timestamp falls within an incident window
    """
    windows_by_series = load_incident_windows(labels_path)

    parts = []
    for csv_path in series_paths:
        series_id = f"{csv_path.parent.name}/{csv_path.name}"

        windows = windows_by_series[series_id]

        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["value"] = pd.to_numeric(df["value"])
        df["series_id"] = series_id

        df["is_incident"] = 0
        for start, end in windows:
            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            df.loc[mask, "is_incident"] = 1

        parts.append(df)

    dataset: pd.DataFrame = pd.concat(parts, ignore_index=True)
    dataset = dataset.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    return dataset


def summarize_series(dataset: pd.DataFrame) -> pd.DataFrame:
    summary = (
        dataset.groupby("series_id", as_index=False)
        .agg(
            rows=("timestamp", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            incident_points=("is_incident", "sum"),
            min_value=("value", "min"),
            max_value=("value", "max"),
        )
        .sort_values("series_id")
        .reset_index(drop=True)
    )
    return summary


def normalize_series(dataset: pd.DataFrame) -> pd.DataFrame:
    """Normalize `value` per series with z-score normalization."""
    dataset = dataset.copy()

    for series_id in dataset["series_id"].unique():
        mask = dataset["series_id"] == series_id
        values = dataset.loc[mask, "value"]
        mean = values.mean()
        std = values.std()
        if std > 0:
            dataset.loc[mask, "value"] = (values - mean) / std
        else:
            dataset.loc[mask, "value"] = 0.0

    return dataset


def make_sliding_windows(
    dataset: pd.DataFrame,
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window samples for incident classification.

    Returns:
    - X: array of shape (n_samples, 1, window_size)
    - y: array of shape (n_samples,) with binary labels (1 if incident in horizon)
    """
    x_samples = []
    y_samples = []

    for _, series_df in dataset.groupby("series_id", sort=False):
        series_df = series_df.sort_values("timestamp").reset_index(drop=True)

        values = series_df["value"].to_numpy(dtype=float)
        labels = series_df["is_incident"].to_numpy()

        n_samples = len(series_df) - window_size - horizon + 1
        for i in range(n_samples):
            x_samples.append(values[i : i + window_size].reshape(1, -1))
            y_samples.append(int(labels[i + window_size : i + window_size + horizon].any()))

    X = np.stack(x_samples) if x_samples else np.empty((0, 1, window_size))
    y = np.array(y_samples, dtype=np.int8)
    return X, y

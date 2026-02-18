from __future__ import annotations

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

        if series_id not in windows_by_series:
            raise KeyError(f"No incident windows for '{series_id}' in {labels_path}")

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
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build sliding-window samples for incident classification.

    Returns:
    - X: array of shape (n_samples, window_size, 1)
    - y: array of shape (n_samples,) with binary labels
    - meta: DataFrame with window timestamps for debugging
    """
    x_samples = []
    y_samples = []
    metadata = []

    for series_id, series_df in dataset.groupby("series_id", sort=False):
        series_df = series_df.sort_values("timestamp").reset_index(drop=True)

        values = series_df["value"].to_numpy(dtype=float)
        labels = series_df["is_incident"].to_numpy()
        timestamps = series_df["timestamp"].to_numpy()

        n_samples = len(series_df) - window_size - horizon + 1
        if n_samples <= 0:
            continue

        for i in range(n_samples):
            x_window = values[i : i + window_size]
            y_window = labels[i + window_size : i + window_size + horizon]

            x_samples.append(x_window.reshape(-1, 1))
            y_samples.append(int(y_window.any()))
            metadata.append(
                {
                    "series_id": series_id,
                    "window_start": timestamps[i],
                    "window_end": timestamps[i + window_size - 1],
                    "horizon_start": timestamps[i + window_size],
                    "horizon_end": timestamps[i + window_size + horizon - 1],
                }
            )

    if x_samples:
        X = np.stack(x_samples)
    else:
        X = np.empty((0, window_size, 1), dtype=float)

    y = np.array(y_samples, dtype=np.int8)
    meta = pd.DataFrame(metadata)
    return X, y, meta

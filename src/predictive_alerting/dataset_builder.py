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

    Returns a DataFrame with columns:\n
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


def make_incident_windows(
    dataset: pd.DataFrame,
    window_size: int,
    horizon: int,
    dropna: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build sliding-window samples for incident classification.

    For each position in the series:
    - X uses the previous `window_size` timesteps
    - y is 1 if any incident occurs in the next `horizon` timesteps

    Returns:
        X: array of shape (n_samples, window_size, 1) with metric values
        y: array of shape (n_samples,) with binary labels
        meta: DataFrame with window timestamps for debugging
    """
    x_samples = []
    y_samples = []
    metadata = []

    for series_id, series_df in dataset.groupby("series_id", sort=False):
        series_df = series_df.sort_values("timestamp").reset_index(drop=True)

        values = series_df["value"].to_numpy(dtype=float)
        labels = series_df["is_incident"].to_numpy()
        timestamps = series_df["timestamp"].to_numpy()

        # We need window_size + horizon points to create one sample
        n_samples = len(series_df) - window_size - horizon + 1
        if n_samples <= 0:
            continue

        for i in range(n_samples):
            x_window = values[i : i + window_size]
            y_window = labels[i + window_size : i + window_size + horizon]

            if dropna and (np.isnan(x_window).any() or np.isnan(y_window).any()):
                continue

            x_samples.append(x_window.reshape(-1, 1))  # Shape: (window_size, 1)
            y_samples.append(int(y_window.any()))
            metadata.append({
                "series_id": series_id,
                "window_start": timestamps[i],
                "window_end": timestamps[i + window_size - 1],
                "horizon_start": timestamps[i + window_size],
                "horizon_end": timestamps[i + window_size + horizon - 1],
            })

    if x_samples:
        X = np.stack(x_samples)
    else:
        X = np.empty((0, window_size, 1), dtype=float)

    y = np.array(y_samples, dtype=np.int8)
    meta = pd.DataFrame(metadata)
    return X, y, meta


def build_window_features(windows: np.ndarray) -> pd.DataFrame:
    """
    Compute statistical features from each window.

    Takes windows of shape (n_samples, window_size, 1) and returns a DataFrame
    with one row per window containing summary statistics.

    These features capture the behavior of the metric during the lookback period
    without leaking future information.
    """
    # Squeeze out the last dimension since we only have one feature (value)
    # Shape becomes: (n_samples, window_size)
    w = windows.squeeze(axis=2)

    n_samples, window_size = w.shape
    if n_samples == 0:
        return pd.DataFrame()

    # Basic statistics
    features = pd.DataFrame({
        "value_mean": w.mean(axis=1),
        "value_std": w.std(axis=1, ddof=0),
        "value_min": w.min(axis=1),
        "value_max": w.max(axis=1),
        "value_last": w[:, -1],
        "value_first": w[:, 0],
        "value_diff_last_first": w[:, -1] - w[:, 0],
        "value_median": np.median(w, axis=1),
        "value_q25": np.quantile(w, 0.25, axis=1),
        "value_q75": np.quantile(w, 0.75, axis=1),
    })

    # Slope: linear regression coefficient over the window
    # Using simple formula: slope = sum((t - t_mean) * (x - x_mean)) / sum((t - t_mean)^2)
    t = np.arange(window_size, dtype=float)
    t_centered = t - t.mean()
    denom = np.dot(t_centered, t_centered)
    if denom > 0:
        features["value_slope"] = np.dot(w, t_centered) / denom
    else:
        features["value_slope"] = 0.0

    return features


def build_training_table(
    dataset: pd.DataFrame,
    window_size: int,
    horizon: int,
) -> pd.DataFrame:
    """
    Build a training table with features and labels for incident prediction.

    Creates sliding windows over each series, computes statistical features
    from each window, and labels each sample based on whether an incident
    occurs in the following horizon.
    """
    X, y, meta = make_incident_windows(dataset, window_size, horizon)
    features = build_window_features(X)

    # Combine metadata, features, and label into one table
    result = pd.concat([meta.reset_index(drop=True), features], axis=1)
    result["label"] = y
    return result


def save_training_dataset_parquet(training_df: pd.DataFrame, output_path: Path) -> Path:
    """Save training data to parquet format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(output_path, index=False)
    return output_path

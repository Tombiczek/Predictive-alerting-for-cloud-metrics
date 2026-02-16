from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _resolve_path(path: str | Path, data_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return data_dir / candidate


def _series_key(series_path: Path, data_dir: Path) -> str:
    try:
        return series_path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return series_path.as_posix()


def _load_windows(
    labels_path: Path,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    raw_windows = json.loads(labels_path.read_text())
    windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for key, values in raw_windows.items():
        parsed = [
            (pd.Timestamp(start), pd.Timestamp(end))
            for start, end in values
        ]
        windows[key] = parsed
    return windows


def _label_series(
    df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    labeled = df.copy()
    labeled["is_incident"] = 0
    labeled["incident_window_id"] = -1

    for window_id, (start, end) in enumerate(windows):
        in_window = (labeled["timestamp"] >= start) & (labeled["timestamp"] <= end)
        labeled.loc[in_window, "is_incident"] = 1
        labeled.loc[in_window, "incident_window_id"] = window_id

    labeled["is_incident"] = labeled["is_incident"].astype("int8")
    labeled["incident_window_id"] = labeled["incident_window_id"].astype("int16")
    return labeled


def build_labeled_dataset(
    series_files: Iterable[str | Path],
    labels_path: str | Path,
    data_dir: str | Path = "data",
    strict_labels: bool = True,
) -> pd.DataFrame:
    """Build one long-form DataFrame from multiple metric CSV files.

    The output is grouped by `series_id` and contains:
    `timestamp`, `value`, `is_incident`, and `incident_window_id`.
    """
    data_dir_path = Path(data_dir)
    labels_path_obj = Path(labels_path)
    windows_by_series = _load_windows(labels_path_obj)

    parts: list[pd.DataFrame] = []
    for series_file in series_files:
        csv_path = _resolve_path(series_file, data_dir_path)
        key = _series_key(csv_path, data_dir_path)
        if strict_labels and key not in windows_by_series:
            msg = (
                f"No incident windows found for series key '{key}'. "
                f"Check labels file: {labels_path_obj}"
            )
            raise KeyError(msg)
        windows = windows_by_series.get(key, [])

        part = pd.read_csv(csv_path)
        part["timestamp"] = pd.to_datetime(part["timestamp"])
        part["value"] = pd.to_numeric(part["value"], errors="coerce")
        part["series_id"] = key
        part["file_name"] = csv_path.name
        part = _label_series(part, windows)

        parts.append(part)

    dataset = pd.concat(parts, ignore_index=True)
    dataset = dataset.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    dataset["step"] = dataset.groupby("series_id").cumcount()
    return dataset


def summarize_series(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return per-series sanity metrics for quick EDA checks."""
    summary = (
        dataset.groupby("series_id", as_index=False)
        .agg(
            rows=("timestamp", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            incident_points=("is_incident", "sum"),
            incident_windows=("incident_window_id", lambda x: x[x >= 0].nunique()),
            min_value=("value", "min"),
            max_value=("value", "max"),
            missing_values=("value", lambda x: int(x.isna().sum())),
        )
        .sort_values("series_id")
        .reset_index(drop=True)
    )
    summary["incident_points"] = summary["incident_points"].astype("int64")
    summary["incident_windows"] = summary["incident_windows"].astype("int64")
    return summary


def make_incident_windows(
    dataset: pd.DataFrame,
    window_size: int,
    horizon: int,
    feature_columns: tuple[str, ...] = ("value",),
    label_column: str = "is_incident",
    series_column: str = "series_id",
    timestamp_column: str = "timestamp",
    dropna: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build sliding-window samples for incident classification.

    For each series and each start index ``s``:
    - ``X[s]`` uses steps ``[s, s + window_size)``
    - ``y[s]`` is 1 if any incident appears in ``[s + window_size, s + window_size + horizon)``
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")

    required_columns = {series_column, timestamp_column, label_column, *feature_columns}
    missing = required_columns.difference(dataset.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Dataset is missing required columns: {missing_str}")

    ordered = dataset.sort_values([series_column, timestamp_column]).reset_index(drop=True)

    x_samples: list[np.ndarray] = []
    y_samples: list[int] = []
    metadata: list[dict[str, object]] = []

    for series_id, series_df in ordered.groupby(series_column, sort=False):
        values = series_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
        labels = pd.to_numeric(series_df[label_column], errors="coerce").to_numpy()
        timestamps = pd.to_datetime(series_df[timestamp_column]).to_numpy()

        max_start = len(series_df) - window_size - horizon + 1
        if max_start <= 0:
            continue

        for start in range(max_start):
            history_end = start + window_size
            horizon_end = history_end + horizon

            x_window = values[start:history_end]
            y_window = labels[history_end:horizon_end]

            if dropna and (np.isnan(x_window).any() or np.isnan(y_window).any()):
                continue

            y_value = int((y_window > 0).any())
            x_samples.append(x_window)
            y_samples.append(y_value)
            metadata.append(
                {
                    "series_id": series_id,
                    "window_start": timestamps[start],
                    "window_end": timestamps[history_end - 1],
                    "horizon_start": timestamps[history_end],
                    "horizon_end": timestamps[horizon_end - 1],
                    "window_size": window_size,
                    "horizon": horizon,
                }
            )

    n_features = len(feature_columns)
    if x_samples:
        x_array = np.stack(x_samples)
    else:
        x_array = np.empty((0, window_size, n_features), dtype=float)

    y_array = np.array(y_samples, dtype=np.int8)
    metadata_df = pd.DataFrame(metadata)
    return x_array, y_array, metadata_df


def _window_feature_columns(feature_columns: tuple[str, ...]) -> list[str]:
    columns: list[str] = []
    for feature_name in feature_columns:
        columns.extend(
            [
                f"{feature_name}_mean",
                f"{feature_name}_std",
                f"{feature_name}_min",
                f"{feature_name}_max",
                f"{feature_name}_last",
                f"{feature_name}_first",
                f"{feature_name}_diff_last_first",
                f"{feature_name}_slope",
                f"{feature_name}_median",
                f"{feature_name}_q25",
                f"{feature_name}_q75",
            ]
        )
    return columns


def build_window_features(
    windows_x: np.ndarray,
    feature_columns: tuple[str, ...] = ("value",),
) -> pd.DataFrame:
    """Compute tabular features from history windows only.

    This function intentionally uses only ``windows_x`` (the historical part), so it
    does not leak future information into training features.
    """
    if windows_x.ndim != 3:
        raise ValueError(f"windows_x must be 3D (n_samples, window_size, n_features), got {windows_x.ndim}D")

    n_samples, window_size, n_features = windows_x.shape
    if n_features != len(feature_columns):
        msg = (
            f"Number of feature columns ({len(feature_columns)}) must match "
            f"windows_x.shape[2] ({n_features})."
        )
        raise ValueError(msg)

    output_columns = _window_feature_columns(feature_columns)
    if n_samples == 0:
        return pd.DataFrame(columns=output_columns)

    means = windows_x.mean(axis=1)
    stds = windows_x.std(axis=1, ddof=0)
    mins = windows_x.min(axis=1)
    maxs = windows_x.max(axis=1)
    lasts = windows_x[:, -1, :]
    firsts = windows_x[:, 0, :]
    diffs = lasts - firsts
    medians = np.median(windows_x, axis=1)
    q25 = np.quantile(windows_x, q=0.25, axis=1)
    q75 = np.quantile(windows_x, q=0.75, axis=1)

    time_idx = np.arange(window_size, dtype=float)
    centered_idx = time_idx - time_idx.mean()
    denom = float(np.dot(centered_idx, centered_idx))
    if denom == 0.0:
        slopes = np.zeros_like(means)
    else:
        slopes = np.tensordot(windows_x, centered_idx, axes=(1, 0)) / denom

    feature_values: dict[str, np.ndarray] = {}
    for i, feature_name in enumerate(feature_columns):
        feature_values[f"{feature_name}_mean"] = means[:, i]
        feature_values[f"{feature_name}_std"] = stds[:, i]
        feature_values[f"{feature_name}_min"] = mins[:, i]
        feature_values[f"{feature_name}_max"] = maxs[:, i]
        feature_values[f"{feature_name}_last"] = lasts[:, i]
        feature_values[f"{feature_name}_first"] = firsts[:, i]
        feature_values[f"{feature_name}_diff_last_first"] = diffs[:, i]
        feature_values[f"{feature_name}_slope"] = slopes[:, i]
        feature_values[f"{feature_name}_median"] = medians[:, i]
        feature_values[f"{feature_name}_q25"] = q25[:, i]
        feature_values[f"{feature_name}_q75"] = q75[:, i]

    return pd.DataFrame(feature_values, columns=output_columns)


def build_training_table(
    dataset: pd.DataFrame,
    window_size: int,
    horizon: int,
    feature_columns: tuple[str, ...] = ("value",),
    label_name: str = "label",
    label_column: str = "is_incident",
    series_column: str = "series_id",
    timestamp_column: str = "timestamp",
    include_metadata: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build one tabular dataset with engineered features (X) and binary label (y)."""
    windows_x, y, windows_meta = make_incident_windows(
        dataset=dataset,
        window_size=window_size,
        horizon=horizon,
        feature_columns=feature_columns,
        label_column=label_column,
        series_column=series_column,
        timestamp_column=timestamp_column,
        dropna=dropna,
    )
    features_df = build_window_features(windows_x=windows_x, feature_columns=feature_columns)

    parts = [features_df]
    if include_metadata:
        parts = [windows_meta.reset_index(drop=True), features_df]

    training_df = pd.concat(parts, axis=1)
    training_df[label_name] = y.astype("int8")
    return training_df


def save_training_dataset_parquet(
    training_df: pd.DataFrame,
    output_path: str | Path,
    index: bool = False,
    compression: str = "snappy",
) -> Path:

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(output_path_obj, index=index, compression=compression)

    return output_path_obj

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

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

import numpy as np
import pandas as pd


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

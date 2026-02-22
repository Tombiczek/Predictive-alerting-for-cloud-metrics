import numpy as np
import pandas as pd


def make_sliding_windows(
    dataset: pd.DataFrame,
    window_size: int,
    horizon: int
):
    """
    Build sliding-window samples for binary event within horizon prediction
    based on point anomaly timestamps

    Label definition for each window ending at t:
    - y=1 if there exists an event index e such that:
        t + 1 <= e <= t + horizon AND the history window does not contain an event:
        events[i : i + window_size].any() == False
    - y=0 otherwise.

    Returns:
    - X: array of shape (n_samples, 1, window_size)
    - y: array of shape (n_samples,) with binary labels
    """
    x_samples = []
    y_samples = []
    meta = []

    for series_id, series_df in dataset.groupby("series_id", sort=False):
        series_df = series_df.sort_values("timestamp").reset_index(drop=True)

        values = series_df["value"].to_numpy(dtype=float)
        labels = series_df["is_incident"].to_numpy()
        ts = series_df["timestamp"].to_numpy()

        start = np.zeros_like(labels)
        start[0] = labels[0]
        start[1:] = (labels[1:] == 1) & (labels[:-1] == 0)

        unsafe = labels.astype(bool)

        n = len(series_df)
        for t in range(window_size - 1, n - horizon):
            history_start = t - window_size + 1
            history_end_exclusive = t + 1

            # We keep only clean pre incident history windows
            if unsafe[history_start:history_end_exclusive].any():
                continue

            future_start = t + 1
            future_end_exclusive = t + horizon + 1

            y = int(start[future_start:future_end_exclusive].any())
            x = values[history_start:history_end_exclusive].reshape(1, -1)

            x_samples.append(x)
            y_samples.append(y)
            meta.append({
                "series_id": series_id,
                "t_end_idx": t,
                "t_end_ts": pd.Timestamp(ts[t]),
            })

    X = np.stack(x_samples) if x_samples else np.empty((0, 1, window_size))
    y = np.array(y_samples, dtype=np.int8)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df

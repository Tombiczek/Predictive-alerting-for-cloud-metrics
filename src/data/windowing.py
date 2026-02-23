import numpy as np
import pandas as pd


def make_sliding_windows(
    dataset: pd.DataFrame,
    window_size: int,
    horizon: int
):
    """
    Label definition for each window ending at t:
    - history: [t-window_size+1, t]
    - y=1 if any incident start in (t, t+horizon]
    - skip if history overlaps incident
    """
    x_samples = []
    y_samples = []
    meta = []

    for series_id, series_df in dataset.groupby("series_id", sort=False):
        series_df = series_df.sort_values("timestamp").reset_index(drop=True)

        values = series_df["value"].to_numpy(dtype=float)
        labels = series_df["is_incident"].to_numpy()
        ts = series_df["timestamp"].to_numpy()

        incident_start = np.zeros_like(labels)
        incident_start[0] = labels[0]
        incident_start[1:] = (labels[1:] == 1) & (labels[:-1] == 0)

        in_incident = labels.astype(bool)

        for t in range(window_size - 1, len(series_df) - horizon):
            history_start = t - window_size + 1
            history_end_exclusive = t + 1

            # We keep only clean pre incident history windows
            if in_incident[history_start:history_end_exclusive].any():
                continue

            future_start = t + 1
            future_end_exclusive = t + horizon + 1

            y = int(incident_start[future_start:future_end_exclusive].any())
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

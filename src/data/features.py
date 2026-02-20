import numpy as np


def extract_features(X_windows: np.ndarray) -> np.ndarray:
    """
    Convert raw sliding windows (N, 1, W) to tabular features (N, n_features).
    """
    X = X_windows[:, 0, :]  # (N, W)
    return np.column_stack([
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
        X[:, -1] - X[:, 0],
        np.percentile(X, 90, axis=1),
        np.percentile(X, 10, axis=1),
        (np.diff(X, axis=1) ** 2).mean(axis=1),
        np.median(X, axis=1),
        np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1),
    ])

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tsai.tslearner import TSClassifier


def predict_proba(clf: TSClassifier, X: np.ndarray) -> np.ndarray:
    """
    Run inference through a trained TSClassifier and return probabilities.

    Args:
        clf: A trained TSClassifier (returned by train_model).
        X:   Features, shape (n_samples, 1, window_size).

    Returns:
        Array of probabilities, shape (n_samples,).
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # get_X_preds returns (predictions, targets, probabilities)
    _, _, probs = clf.get_X_preds(X_tensor)
    return probs.numpy().flatten()


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Find the classification threshold that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else float(thresholds[-1])


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> dict:
    """Compute binary classification metrics at a given threshold."""
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def evaluate_model(
    clf: TSClassifier,
    X: np.ndarray,
    y: np.ndarray,
    prefix: str = "val",
) -> dict:
    """
    Evaluate a trained TSClassifier, find the optimal threshold, and log to wandb.

    Args:
        clf:    Trained TSClassifier (returned by train_model).
        X:      Features, shape (n_samples, 1, window_size).
        y:      Ground-truth labels, shape (n_samples,).
        prefix: Metric name prefix logged to wandb (e.g. "val", "test").

    Returns:
        Dictionary of computed metrics.
    """
    y_probs = predict_proba(clf, X)
    threshold = find_optimal_threshold(y, y_probs)
    metrics = compute_metrics(y, y_probs, threshold)

    # Log to wandb with prefix
    wandb.log({f"{prefix}_{k}": v for k, v in metrics.items()})

    print(
        f"{prefix.capitalize()} metrics: "
        f"acc={metrics['accuracy']:.4f}, "
        f"prec={metrics['precision']:.4f}, "
        f"rec={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"auc={metrics['roc_auc']:.4f}, "
        f"threshold={metrics['threshold']:.4f}"
    )

    return metrics


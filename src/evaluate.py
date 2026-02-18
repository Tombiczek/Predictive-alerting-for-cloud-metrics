import numpy as np
import torch
import torch.nn as nn
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


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Run inference and return predicted probabilities.

    Args:
        model: Trained PyTorch model
        X: Features, shape (n_samples, 1, window_size)

    Returns:
        Array of probabilities, shape (n_samples,)
    """
    device = next(model.parameters()).device
    model.eval()

    X_t = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        outputs = model(X_t)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

    return probs


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Find optimal classification threshold by maximizing F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else float(thresholds[-1])


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> dict:
    """Compute classification metrics at a given threshold."""
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
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    prefix: str = "val",
) -> dict:
    """
    Evaluate model, find optimal threshold, and log metrics to wandb.

    Args:
        model: Trained model
        X: Features
        y: Labels
        prefix: Metric name prefix (e.g., "val", "test")

    Returns:
        Dictionary of metrics
    """
    y_probs = predict_proba(model, X)
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


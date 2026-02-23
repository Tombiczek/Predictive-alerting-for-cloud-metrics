import os
import pickle
from pathlib import Path

import wandb
from dotenv import load_dotenv

from src.data.datasets import load_features_dataset
from src.evaluate import alerting_eval, pick_threshold, predict_proba_sklearn
from src.train import train_tree_classifier

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 12
HORIZON = 12

# Configuration taken from wandb sweep
CONFIG = {
    "model_name": "RandomForest",
    "window_size": 12,
    "horizon": 12,
    "n_estimators": 300,
    "criterion": "entropy",
    "max_features": "sqrt",
    "max_depth": 50,
    "max_leaf_nodes": 1000,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "min_impurity_decrease": 0.0,
    "ccp_alpha": 0.0,
    "bootstrap": True,
    "class_weight": "balanced",
}


def main():
    wandb.init(
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics",
        config=CONFIG,
        name="rf_best_2",
    )

    data = load_features_dataset(WINDOW_SIZE, HORIZON)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    meta_val = data["meta_val"]
    meta_test = data["meta_test"]
    incident_windows = data["incident_windows_by_series"]

    print("\nTraining model...")
    model = train_tree_classifier(X_train, y_train, CONFIG)

    print("\nEvaluating on validation set...")
    val_probs = predict_proba_sklearn(model, X_val)
    val_metrics = pick_threshold(
        meta_val=meta_val,
        probs_val=val_probs,
        incident_windows_by_series=incident_windows,
        horizon_steps=HORIZON,
    )

    best_threshold = val_metrics["threshold"]

    print("\nEvaluating on test set...")
    test_probs = predict_proba_sklearn(model, X_test)
    test_metrics = alerting_eval(
        meta_df=meta_test,
        y_probs=test_probs,
        incident_windows_by_series=incident_windows,
        threshold=best_threshold,
        horizon_steps=HORIZON,
    )

    wandb.log(
        {
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
    )

    model_path = ARTIFACTS_DIR / "rf_best_2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "config": CONFIG}, f)

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()

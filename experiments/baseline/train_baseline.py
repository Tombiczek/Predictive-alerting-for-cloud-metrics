import os
import pickle
from pathlib import Path

import wandb
from dotenv import load_dotenv

from src.data.datasets import load_features_dataset
from src.evaluate import evaluate_model, predict_proba_sklearn
from src.train import train_tree_classifier

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 24
HORIZON = 12

# Configuration taken from wandb sweep
CONFIG = {
    "model_name": "RandomForest",
    "window_size": 24,
    "horizon": 12,
    "n_estimators": 300,
    "max_features": "log2",
    "max_depth": 5,
    "max_leaf_nodes": 100,
    "min_samples_split": 10,
    "min_samples_leaf": 2,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced",
}


def main():
    wandb.init(
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics",
        config=CONFIG,
        name="rf_best"
    )

    data = load_features_dataset(WINDOW_SIZE, HORIZON)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("\nTraining model...")
    model = train_tree_classifier(X_train, y_train, CONFIG)

    print("\nEvaluating on validation set...")
    val_probs = predict_proba_sklearn(model, X_val)
    val_metrics = evaluate_model(y_val, val_probs, prefix="val")

    print("\nEvaluating on test set...")
    test_probs = predict_proba_sklearn(model, X_test)
    test_metrics = evaluate_model(y_test, test_probs, prefix="test", threshold=val_metrics["threshold"])

    wandb.log({
        **val_metrics,
        **test_metrics
    })

    model_path = ARTIFACTS_DIR / "rf_best.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "config": CONFIG}, f)

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()

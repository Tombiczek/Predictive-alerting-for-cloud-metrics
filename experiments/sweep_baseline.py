import argparse

import wandb

from src.data.datasets import load_features_dataset
from src.train import train_sklearn_model

WINDOW_SIZE = 24
HORIZON = 12

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "val_average_precision"},
    "parameters": {
        "n_estimators": {"values": [100, 300, 500]},
        "max_features": {"values": ["sqrt", "log2"]},
        "max_depth": {"values": [None, 10, 20]},
        "min_samples_split": {"values": [2, 5]},
        "min_samples_leaf": {"values": [1, 2]},
        "class_weight": {"value": "balanced"},
    },
}


data = load_features_dataset(WINDOW_SIZE, HORIZON)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]


def run_sweep_trial():
    wandb.init(
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics"
    )

    config = {
        "model_name": "RandomForest",
        "window_size": WINDOW_SIZE,
        "horizon": HORIZON,
        "n_estimators": wandb.config.n_estimators,
        "max_features": wandb.config.max_features,
        "max_depth": wandb.config.max_depth,
        "min_samples_split": wandb.config.min_samples_split,
        "min_samples_leaf": wandb.config.min_samples_leaf,
        "class_weight": wandb.config.class_weight,
    }

    clf = train_sklearn_model(X_train, y_train, config)

    val_probs = clf.predict_proba(X_val)[:, 1]
    from sklearn.metrics import average_precision_score
    val_ap = average_precision_score(y_val, val_probs)

    wandb.log({"val_average_precision": val_ap})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep for hyperparameter tuning.")
    parser.add_argument(
        "--count",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        sweep_configuration,
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics"
    )
    print(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, function=run_sweep_trial, count=args.count)

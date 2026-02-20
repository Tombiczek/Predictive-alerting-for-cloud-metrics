import argparse

import wandb
from sklearn.metrics import average_precision_score

from src.data.datasets import load_features_dataset
from src.evaluate import predict_proba_sklearn
from src.train import train_tree_classifier

WINDOW_SIZE = 24
HORIZON = 12

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_average_precision"},
    "parameters": {
        "n_estimators": {"values": [100, 300, 500]},
        "max_features": {"values": ["sqrt", None]},
        "max_depth": {"values": [None, 10, 20]},
        "max_leaf_nodes": {"values": [None, 100, 500]},
        "min_samples_split": {"values": [2, 5]},
        "min_samples_leaf": {"values": [1, 2, 5]},
        "min_impurity_decrease": {"value": 0.0},
        "class_weight": {"value": "balanced"},
        "ccp_alpha": {"values": [0.0, 0.0001, 0.001, 0.01]},
        "bootstrap": {"values": [True, False]},
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
        "max_leaf_nodes": wandb.config.max_leaf_nodes,
        "min_samples_split": wandb.config.min_samples_split,
        "min_samples_leaf": wandb.config.min_samples_leaf,
        "min_impurity_decrease": wandb.config.min_impurity_decrease,
        "class_weight": wandb.config.class_weight,
    }

    clf = train_tree_classifier(X_train, y_train, config)

    val_probs = predict_proba_sklearn(clf, X_val)
    train_probs = predict_proba_sklearn(clf, X_train)

    wandb.log({
    "val_average_precision": average_precision_score(y_val, val_probs),
    "train_average_precision": average_precision_score(y_train, train_probs)
    })

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        sweep_configuration,
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics"
    )
    print(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, function=run_sweep_trial, count=args.count)

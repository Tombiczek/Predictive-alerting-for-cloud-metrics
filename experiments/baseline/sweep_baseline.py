import argparse
import os

import wandb
from dotenv import load_dotenv

from src.data.datasets import load_features_dataset
from src.evaluate import predict_proba_sklearn, pick_threshold
from src.train import train_tree_classifier

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

WINDOW_SIZE = 12
HORIZON = 12

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val_alert_score"},
    "parameters": {
        "n_estimators": {"values": [100, 300, 500, 800]},
        "max_features": {"values": ["sqrt", "log2", 0.5]},
        "max_depth": {"values": [5, 10, 20, 30, 50]},
        "max_leaf_nodes": {"values": [50, 100, 500, 1000]},
        "min_samples_split": {"values": [2, 5, 10]},
        "min_samples_leaf": {"values": [1, 2, 5, 10]},
        "min_impurity_decrease": {"value": 0.0},
        "class_weight": {"value": "balanced"},
        "ccp_alpha": {"values": [0.0, 0.0001, 0.001, 0.01]},
        "bootstrap": {"values": [True, False]},
        "criterion": {"values": ["gini", "entropy"]},
    },
}

data = load_features_dataset(WINDOW_SIZE, HORIZON)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
meta_val = data["meta_val"]
incident_windows_by_series = data["incident_windows_by_series"]


def run_sweep_trial():
    wandb.init()

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
        "bootstrap": wandb.config.bootstrap,
        "ccp_alpha": wandb.config.ccp_alpha,
        "criterion": wandb.config.criterion,
    }

    clf = train_tree_classifier(X_train, y_train, config)

    val_probs = predict_proba_sklearn(clf, X_val)

    best = pick_threshold(
        meta_val=meta_val,
        probs_val=val_probs,
        incident_windows_by_series=incident_windows_by_series,
        horizon_steps=HORIZON,
    )
    val_alert_score = best["incident_recall"] - 0.01 * best["false_alerts_per_day"]

    wandb.log(
        {
            "val_incident_recall": best["incident_recall"],
            "val_false_alerts_per_day": best["false_alerts_per_day"],
            "val_alert_score": val_alert_score,
            "val_threshold": best["threshold"],
            "val_incidents_total": best["incidents_total"],
            "val_incidents_caught": best["incidents_caught"],
            "val_lead_time_median_min": best.get("lead_time_median_min"),
        }
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        type=int,
        default=80,
    )
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        sweep_configuration,
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics",
    )
    print(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, function=run_sweep_trial, count=args.count)

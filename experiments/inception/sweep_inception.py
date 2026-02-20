import argparse
import os

import wandb
from dotenv import load_dotenv

from src.data.datasets import load_timeseries_dataset
from src.train import train_deep_classifier

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

WINDOW_SIZE = 24
HORIZON = 12

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "valid_loss"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 5e-4,
            "max": 2e-3,
        },
        "batch_size": {"values": [32, 64]},
        "pos_weight": {
            "distribution": "log_uniform_values",
            "min": 4.0,
            "max": 10.0,
        },
        "nf": {"values": [16, 32]},
        "depth": {"values": [3, 6]},
        "ks": {"values": [11, 23]},
        "epochs": {"value": 50},
        "patience": {"value": 10},
    },
}

data = load_timeseries_dataset(WINDOW_SIZE, HORIZON)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

def run_sweep_trial():
    wandb.init(
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics"
    )

    config = {
        "model_name": "InceptionTimePlus",
        "window_size": WINDOW_SIZE,
        "horizon": HORIZON,
        "batch_size": wandb.config.batch_size,
        "learning_rate": wandb.config.learning_rate,
        "epochs": wandb.config.epochs,
        "patience": wandb.config.patience,
        "pos_weight": wandb.config.pos_weight,
        "model_kwargs": {
            "nf": wandb.config.nf,
            "depth": wandb.config.depth,
            "ks": wandb.config.ks,
        },
    }

    train_deep_classifier(X_train, y_train, X_val, y_val, config)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        type=int,
        default=20
    )
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        sweep_configuration,
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics"
    )
    print(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, function=run_sweep_trial, count=args.count)

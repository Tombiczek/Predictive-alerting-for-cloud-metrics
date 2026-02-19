import argparse
from pathlib import Path

import wandb

from src.data import build_labeled_dataset, make_sliding_windows, normalize_series
from src.train import train_model

DATA_DIR = Path("data")
LABELS_PATH = DATA_DIR / "combined_windows.json"

TRAIN_SERIES = [
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_network_in_257a54.csv",
    DATA_DIR / "realAWSCloudwatch/grok_asg_anomaly.csv",
    DATA_DIR / "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
    DATA_DIR / "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
]

VAL_SERIES = [
    DATA_DIR / "realAWSCloudwatch/ec2_network_in_5abac7.csv",
    DATA_DIR / "realAWSCloudwatch/elb_request_count_8c0756.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv",
]

TEST_SERIES = [
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv",
    DATA_DIR / "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv",
]

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


print("Loading data...")
train_df = build_labeled_dataset(TRAIN_SERIES, LABELS_PATH)
train_df = normalize_series(train_df)
X_train, y_train = make_sliding_windows(train_df, WINDOW_SIZE, HORIZON)
print(f"Training set: {X_train.shape}, positive ratio: {y_train.mean():.4f}")

val_df = build_labeled_dataset(VAL_SERIES, LABELS_PATH)
val_df = normalize_series(val_df)
X_val, y_val = make_sliding_windows(val_df, WINDOW_SIZE, HORIZON)
print(f"Validation set: {X_val.shape}, positive ratio: {y_val.mean():.4f}")

test_df = build_labeled_dataset(TEST_SERIES, LABELS_PATH)
test_df = normalize_series(test_df)
X_test, y_test = make_sliding_windows(test_df, WINDOW_SIZE, HORIZON)
print(f"Test set: {X_test.shape}, positive ratio: {y_test.mean():.4f}")


def run_sweep_trial():
    wandb.init(
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics"
    )

    config = {
        "model_name": "InceptionTime",
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

    train_model(X_train, y_train, X_val, y_val, config)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep for hyperparameter tuning.")
    parser.add_argument(
        "--count",
        type=int,
        default=30
    )
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        sweep_configuration,
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics"
    )
    print(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, function=run_sweep_trial, count=args.count)

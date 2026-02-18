"""
Example training script for predictive alerting.

Run from project root:
    uv run python example_train.py
"""
from pathlib import Path

import torch
import wandb

from src.data import build_labeled_dataset, make_sliding_windows, normalize_series
from src.evaluate import evaluate_model
from src.train import train_model

# Data paths
DATA_DIR = Path("data")
LABELS_PATH = DATA_DIR / "combined_windows.json"

TRAIN_SERIES = [
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv",
]

VAL_SERIES = [
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
]

TEST_SERIES = [
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv",
]

# Model configuration
CONFIG = {
    "model_name": "InceptionTime",
    "window_size": 24,
    "horizon": 12,
    "model_kwargs": {},
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "patience": 10,
    "pos_weight": 1.0,
}


def main():
    wandb.init(project="predictive-alerting", config=CONFIG)

    # Load data
    print("Loading training data...")
    train_df = build_labeled_dataset(TRAIN_SERIES, LABELS_PATH)
    train_df = normalize_series(train_df)
    X_train, y_train = make_sliding_windows(train_df, CONFIG["window_size"], CONFIG["horizon"])
    print(f"Training set: {X_train.shape}, positive ratio: {y_train.mean():.4f}")

    print("Loading validation data...")
    val_df = build_labeled_dataset(VAL_SERIES, LABELS_PATH)
    val_df = normalize_series(val_df)
    X_val, y_val = make_sliding_windows(val_df, CONFIG["window_size"], CONFIG["horizon"])
    print(f"Validation set: {X_val.shape}, positive ratio: {y_val.mean():.4f}")

    print("Loading test data...")
    test_df = build_labeled_dataset(TEST_SERIES, LABELS_PATH)
    test_df = normalize_series(test_df)
    X_test, y_test = make_sliding_windows(test_df, CONFIG["window_size"], CONFIG["horizon"])
    print(f"Test set: {X_test.shape}, positive ratio: {y_test.mean():.4f}")

    # Train
    print("\nTraining model...")
    model = train_model(X_train, y_train, X_val, y_val, CONFIG)

    # Evaluate
    print("\nEvaluating on validation set...")
    evaluate_model(model, X_val, y_val, prefix="val")

    print("\nEvaluating on test set...")
    evaluate_model(model, X_test, y_test, prefix="test")

    # Save model artifact to wandb
    model_path = Path("model.pt")
    torch.save({"state_dict": model.state_dict(), "config": CONFIG}, model_path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)
    model_path.unlink()  # Clean up local file

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()



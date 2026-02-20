from pathlib import Path

import torch
import wandb

from src.data.datasets import load_timeseries_dataset
from src.evaluate import evaluate_model, predict_proba_tsai
from src.train import train_deep_classifier

WINDOW_SIZE = 24
HORIZON = 12

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
    data = load_timeseries_dataset(WINDOW_SIZE, HORIZON)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Train
    print("\nTraining model...")
    model = train_deep_classifier(X_train, y_train, X_val, y_val, CONFIG)

    # Evaluate
    print("\nEvaluating on validation set...")
    val_probs = predict_proba_tsai(model, X_val)
    val_metrics = evaluate_model(y_val, val_probs, prefix="val")

    print("\nEvaluating on test set...")
    test_probs = predict_proba_tsai(model, X_test)
    evaluate_model(y_test, test_probs, prefix="test", threshold=val_metrics["threshold"])

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



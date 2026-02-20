import os
import pickle
from pathlib import Path

import wandb
from dotenv import load_dotenv

from src.data.datasets import load_timeseries_dataset
from src.evaluate import evaluate_model, predict_proba_tsai
from src.train import train_deep_classifier

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 24
HORIZON = 12

# Model configuration
CONFIG = {
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


def main():
    wandb.init(
        entity="tombik-warsaw-university-of-technology",
        project="Predictive-alerting-for-cloud-metrics",
        config=CONFIG,
        name="ict_best"
    )

    data = load_timeseries_dataset(WINDOW_SIZE, HORIZON)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("\nTraining model...")
    model = train_deep_classifier(X_train, y_train, X_val, y_val, CONFIG)

    print("\nEvaluating on validation set...")
    val_probs = predict_proba_tsai(model, X_val)
    val_metrics = evaluate_model(y_val, val_probs, prefix="val")

    print("\nEvaluating on test set...")
    test_probs = predict_proba_tsai(model, X_test)
    test_metrics = evaluate_model(y_test, test_probs, prefix="test", threshold=val_metrics["threshold"])

    wandb.log({
        **val_metrics,
        **test_metrics
    })

    model_path = ARTIFACTS_DIR / "icp_best.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "config": CONFIG}, f)

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()

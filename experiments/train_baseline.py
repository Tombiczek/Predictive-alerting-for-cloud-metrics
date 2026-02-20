import pickle
from pathlib import Path

import wandb

from src.data.datasets import load_features_dataset
from src.evaluate import evaluate_model, predict_proba_sklearn
from src.train import train_sklearn_model

WINDOW_SIZE = 24
HORIZON = 12

# Model configuration
CONFIG = {
    "model_name": "RandomForest",
    "window_size": 24,
    "horizon": 12,
    "n_estimators": 100,
    "max_features": "sqrt",
    "max_depth": None,
    "max_leaf_nodes": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_impurity_decrease": 0.0,
    "class_weight": "balanced",
}


def main():
    wandb.init(project="predictive-alerting", config=CONFIG)

    data = load_features_dataset(WINDOW_SIZE, HORIZON)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Train
    print("\nTraining model...")
    model = train_sklearn_model(X_train, y_train, CONFIG)

    # Evaluate
    print("\nEvaluating on validation set...")
    val_probs = predict_proba_sklearn(model, X_val)
    val_metrics = evaluate_model(y_val, val_probs, prefix="val")

    print("\nEvaluating on test set...")
    test_probs = predict_proba_sklearn(model, X_test)
    evaluate_model(y_test, test_probs, prefix="test", threshold=val_metrics["threshold"])

    # Save model artifact to wandb
    model_path = Path("model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "config": CONFIG}, f)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)
    model_path.unlink()  # Clean up local file

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()




import numpy as np
import torch
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.wandb import WandbCallback
from fastai.metrics import APScoreBinary
from sklearn.ensemble import RandomForestClassifier
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.tslearner import TSClassifier


def train_tsai_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> TSClassifier:

    X = np.concatenate([X_train, X_val]).astype(np.float32)
    y = np.concatenate([y_train, y_val]).astype(np.float32).reshape(-1, 1)

    n_train = len(X_train)
    splits = (list(range(n_train)), list(range(n_train, len(X))))

    batch_size = config.get("batch_size", 32)
    lr = config.get("learning_rate", 1e-3)
    epochs = config.get("epochs", 50)
    patience = config.get("patience", 10)
    pos_weight = config.get("pos_weight", 1.0)
    model_kwargs = config.get("model_kwargs", {})

    weight = torch.tensor([pos_weight])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    metrics = [APScoreBinary()]

    callbacks = [
        SaveModelCallback(fname="best"),
        EarlyStoppingCallback(patience=patience),
        WandbCallback(log="all", log_preds=False),
    ]

    clf = TSClassifier(
        X=X,
        y=y,
        splits=splits,
        bs=batch_size,
        arch=InceptionTimePlus,
        arch_config=model_kwargs,
        loss_func=loss_fn,
        lr=lr,
        metrics=metrics,
        cbs=callbacks,
        shuffle_train=True,
    )

    clf.fit(epochs)

    clf.learner.load("best")

    return clf

def train_sklearn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> RandomForestClassifier:

    n_estimators = config.get("n_estimators", 100)
    max_features = config.get("max_features", "sqrt")
    max_depth = config.get("max_depth", None)
    max_leaf_nodes = config.get("max_leaf_nodes", None)
    min_samples_split = config.get("min_samples_split", 2)
    min_samples_leaf = config.get("min_samples_leaf", 1)
    min_impurity_decrease = config.get("min_impurity_decrease", 0.0)
    class_weight = config.get("class_weight", "balanced")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    return clf

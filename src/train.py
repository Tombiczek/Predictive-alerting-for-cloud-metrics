import numpy as np
import torch
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.wandb import WandbCallback
from fastai.metrics import APScoreBinary
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.tslearner import TSClassifier


def train_model(
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

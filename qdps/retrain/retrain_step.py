"""Framework-agnostic retraining primitive: dispatches to a backend.

Keeps the runner (``run_retrain.py``) free of any TF/Torch detail — only the
backends import a deep-learning framework.
"""
from dataclasses import dataclass


@dataclass
class RetrainConfig:
    epochs: int = 30
    batch_size: int = 50
    lr: float = 0.005
    val_size: int = 2500
    loss: str = "categorical_crossentropy"


def _backend(framework):
    if framework == "keras":
        from qdps.retrain.backends import keras_backend
        return keras_backend
    if framework == "torch":
        from qdps.retrain.backends import torch_backend
        return torch_backend
    raise ValueError(f"Unknown framework: {framework}")


def original_accuracy(raw, model_loader, V):
    return _backend(raw.framework).original_accuracy(raw, model_loader, V)


def retrain_and_eval(raw, model_loader, selected, V, cfg, seed):
    return _backend(raw.framework).retrain_and_eval(raw, model_loader, selected, V, cfg, seed)

"""Framework-agnostic retraining primitive: dispatches to a backend.

Keeps the runner (``run_retrain.py``) free of any TF/Torch detail — only the
backends import a deep-learning framework.
"""
from dataclasses import dataclass


@dataclass
class RetrainConfig:
    epochs: int = 30
    batch_size: int = 50
    optimizer: str = "adadelta"     # "adadelta" | "adam"
    lr: float = 0.005
    val_size: int = 2500
    loss: str = "categorical_crossentropy"


# Per-subject deviations from the default protocol, mirroring the original
# scripts: retrain_four.py (default) vs retrain_fruit.py (Adam 1e-5, batch 100,
# sparse integer labels, 5000-sample fit-monitoring split).
SUBJECT_CONFIGS = {
    "Fruit360_ResNet50": RetrainConfig(
        epochs=30, batch_size=100, optimizer="adam", lr=0.00001,
        val_size=5000, loss="sparse_categorical_crossentropy",
    ),
    # retrain_tiny.py: fine-tune from checkpoint, SGD 1e-3 (both parts), batch 64,
    # 10 epochs, CrossEntropy, best-epoch accuracy on V. (val_size unused: the
    # torch backend evaluates directly on V each epoch.)
    "TinyImageNet_ResNet101": RetrainConfig(
        epochs=10, batch_size=64, optimizer="sgd", lr=0.001,
        val_size=0, loss="cross_entropy",
    ),
}


def config_for(subject_key):
    return SUBJECT_CONFIGS.get(subject_key, RetrainConfig())


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

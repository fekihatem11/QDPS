"""Raw images + labels + pretrained model loading for the RQ4 retraining experiment.

This is the package analogue of the SETS ``dataset()`` function in
``SETS/Source_code/retrain_four.py``. Each subject's raw data loader returns a
:class:`RawSubject`; image preprocessing follows the load-bearing [-0.5, 0.5]
convention (``x/255 - 0.5``). Adding a subject = adding one entry to
``RAW_LOADERS`` (and, for non-Keras subjects, a backend).
"""
from dataclasses import dataclass

import numpy as np

from qdps.io.paths import PRETRAINED_MODELS


class MissingDataError(FileNotFoundError):
    """Raised when a subject's raw dataset is not available on disk."""


@dataclass
class RawSubject:
    x_train: np.ndarray      # preprocessed train images
    y_train_oh: np.ndarray   # one-hot train labels (N, n_classes)
    x_test: np.ndarray       # preprocessed test images
    y_test_int: np.ndarray   # integer test labels (n_test,)
    n_classes: int
    framework: str           # "keras" | "torch"


# Pretrained model filename per subject_key (exact on-disk casing).
MODEL_FILE_MAP = {
    "mnist_LeNet1": "model_mnist_LeNet1.h5",
    "mnist_LeNet5": "model_mnist_LeNet5.h5",
    "Fashion_mnist_LeNet4": "model_Fashion_mnist_LeNet4.h5",
    "cifar10_12Conv": "model_cifar10_12Conv.h5",
    "cifar10_ResNet20": "model_cifar10_ResNet20.h5",
    "SVHN_LeNet5": "model_SVHN_LeNet5.h5",
    "Fruit360_ResNet50": "fruit_resnet2.h5",
    "TinyImageNet_ResNet101": "model_best.pth.tar",
}


def _scale(x):
    """SETS image preprocessing: uint8 [0,255] -> float32 [-0.5, 0.5]."""
    return x.astype("float32") / 255.0 - 0.5


# ---- per-dataset raw loaders ----

def _load_mnist():
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = _scale(x_train.reshape(-1, 28, 28, 1))
    x_test = _scale(x_test.reshape(-1, 28, 28, 1))
    return RawSubject(
        x_train=x_train,
        y_train_oh=to_categorical(y_train, 10),
        x_test=x_test,
        y_test_int=np.asarray(y_test).astype(int).ravel(),
        n_classes=10,
        framework="keras",
    )


# subject_key -> callable returning a RawSubject. Phase 1 = MNIST only;
# other subjects are added in later phases (Fashion/CIFAR/SVHN/Fruit/Tiny).
RAW_LOADERS = {
    "mnist_LeNet1": _load_mnist,
    "mnist_LeNet5": _load_mnist,
}


def load_raw_subject(subject_key):
    """Load raw images + labels for a subject."""
    if subject_key not in RAW_LOADERS:
        raise MissingDataError(
            f"No raw-data loader for '{subject_key}' yet. Implemented: "
            f"{sorted(RAW_LOADERS)}"
        )
    return RAW_LOADERS[subject_key]()


def load_pretrained_model(subject_key):
    """Fresh-load the pretrained model for a subject (Keras subjects only here)."""
    if subject_key not in MODEL_FILE_MAP:
        raise KeyError(f"Unknown subject_key: {subject_key}")
    path = PRETRAINED_MODELS / MODEL_FILE_MAP[subject_key]
    if not path.exists():
        raise MissingDataError(
            f"Pretrained model not found: {path}. Vendor it from "
            f"SETS/Input_data/Pretrained_model/{MODEL_FILE_MAP[subject_key]}"
        )
    from tensorflow.keras.models import load_model
    return load_model(str(path), compile=False)

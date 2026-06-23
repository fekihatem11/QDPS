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

def _keras_subject(x_train, y_train, x_test, y_test, n_classes):
    """Build a RawSubject from raw keras arrays with the shared preprocessing."""
    from tensorflow.keras.utils import to_categorical
    return RawSubject(
        x_train=_scale(x_train),
        y_train_oh=to_categorical(y_train, n_classes),
        x_test=_scale(x_test),
        y_test_int=np.asarray(y_test).astype(int).ravel(),
        n_classes=n_classes,
        framework="keras",
    )


def _load_mnist():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return _keras_subject(
        x_train.reshape(-1, 28, 28, 1), y_train,
        x_test.reshape(-1, 28, 28, 1), y_test, 10,
    )


def _load_fashion():
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return _keras_subject(
        x_train.reshape(-1, 28, 28, 1), y_train,
        x_test.reshape(-1, 28, 28, 1), y_test, 10,
    )


def _load_cifar10():
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # cifar images are already (N, 32, 32, 3); labels come as (N, 1).
    return _keras_subject(x_train, y_train, x_test, y_test, 10)


# subject_key -> callable returning a RawSubject. SVHN/Fruit/Tiny are added as
# their raw data + (for Tiny) a torch backend land.
RAW_LOADERS = {
    "mnist_LeNet1": _load_mnist,
    "mnist_LeNet5": _load_mnist,
    "Fashion_mnist_LeNet4": _load_fashion,
    "cifar10_12Conv": _load_cifar10,
    "cifar10_ResNet20": _load_cifar10,
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

"""Raw images + labels + pretrained model loading for the RQ4 retraining experiment.

This is the package analogue of the SETS ``dataset()`` function in
``SETS/Source_code/retrain_four.py``. Each subject's raw data loader returns a
:class:`RawSubject`; image preprocessing follows the load-bearing [-0.5, 0.5]
convention (``x/255 - 0.5``). Adding a subject = adding one entry to
``RAW_LOADERS`` (and, for non-Keras subjects, a backend).
"""
from dataclasses import dataclass

import numpy as np

from qdps.io.paths import PRETRAINED_MODELS, RAW_DATA_DIR


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
    # Index range the T/V split is drawn from. The original retrain_four.py
    # hardcodes range(10000) even for SVHN (26k test images), so V must be the
    # complement of T within the first v_pool_size indices, not the full test set.
    v_pool_size: int = 10000


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


def _load_svhn():
    """SVHN from the Stanford .mat files (labels 1..10, '10' = digit 0).

    Faithful to retrain_four.py: LabelBinarizer one-hot for train; test labels
    one-hot then argmax to ints. The (32,32,3,N) arrays are moved to (N,32,32,3).
    """
    from scipy.io import loadmat
    from sklearn.preprocessing import LabelBinarizer
    svhn_dir = RAW_DATA_DIR / "svhn"
    train_path = svhn_dir / "train_32x32.mat"
    test_path = svhn_dir / "test_32x32.mat"
    if not train_path.exists() or not test_path.exists():
        raise MissingDataError(
            f"SVHN .mat files not found in {svhn_dir}. Download train_32x32.mat "
            f"and test_32x32.mat from http://ufldl.stanford.edu/housenumbers/"
        )
    train_raw = loadmat(str(train_path))
    test_raw = loadmat(str(test_path))
    x_train = np.moveaxis(np.array(train_raw["X"]), -1, 0).reshape(-1, 32, 32, 3)
    x_test = np.moveaxis(np.array(test_raw["X"]), -1, 0).reshape(-1, 32, 32, 3)
    lb = LabelBinarizer()
    y_train_oh = np.asarray(lb.fit_transform(train_raw["y"]))
    y_test_int = np.argmax(np.asarray(lb.fit_transform(test_raw["y"])), axis=1)
    return RawSubject(
        x_train=_scale(x_train),
        y_train_oh=y_train_oh,
        x_test=_scale(x_test),
        y_test_int=y_test_int,
        n_classes=10,
        framework="keras",
    )


def _load_fruit360():
    """Fruit-360 from prebuilt npy arrays (see build_fruit_npy.py).

    Faithful to retrain_fruit.py: images stay UNSCALED (raw 0..255 float32,
    ``flag=False`` in the original — the model expects that), labels are
    INTEGER class indices used with sparse_categorical_crossentropy, and the
    T/V pool spans the full 23619-image test set.
    """
    fruit_dir = RAW_DATA_DIR / "fruit360"
    needed = ["fruit_x_train_origin.npy", "fruit_y_train.npy",
              "fruit_x_test_origin.npy", "fruit_y_test.npy"]
    missing = [n for n in needed if not (fruit_dir / n).exists()]
    if missing:
        raise MissingDataError(
            f"Fruit360 arrays missing in {fruit_dir}: {missing}. Build them with "
            f"qdps/retrain/build_fruit_npy.py from the fruits-360-100x100 dataset."
        )
    x_train = np.load(fruit_dir / "fruit_x_train_origin.npy")
    y_train = np.load(fruit_dir / "fruit_y_train.npy")
    x_test = np.load(fruit_dir / "fruit_x_test_origin.npy")
    y_test = np.load(fruit_dir / "fruit_y_test.npy")
    return RawSubject(
        x_train=x_train,
        y_train_oh=y_train,          # integer labels (sparse CE) — see docstring
        x_test=x_test,
        y_test_int=np.asarray(y_test).astype(int).ravel(),
        n_classes=141,
        framework="keras",
        v_pool_size=len(x_test),     # retrain_fruit.py: elements = range(23619)
    )


# subject_key -> callable returning a RawSubject. TinyImageNet is added when
# its raw data + torch backend land.
RAW_LOADERS = {
    "mnist_LeNet1": _load_mnist,
    "mnist_LeNet5": _load_mnist,
    "Fashion_mnist_LeNet4": _load_fashion,
    "cifar10_12Conv": _load_cifar10,
    "cifar10_ResNet20": _load_cifar10,
    "SVHN_LeNet5": _load_svhn,
    "Fruit360_ResNet50": _load_fruit360,
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

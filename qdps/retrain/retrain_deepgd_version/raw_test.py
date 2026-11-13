"""Raw test images (uint8, [0, 255]) and integer labels, per dataset.

These are the images the transformations are applied to: the full test set of
each dataset, in the same order and with the same labels as the retraining code
(``qdps/retrain/raw_data.py``), before any model preprocessing.
"""
import os

import numpy as np

from qdps.io.paths import RAW_DATA_DIR
from qdps.retrain.raw_data import MissingDataError


def _mnist():
    from tensorflow.keras.datasets import mnist
    (_, _), (x, y) = mnist.load_data()
    return x.reshape(-1, 28, 28, 1), y


def _fashion():
    from tensorflow.keras.datasets import fashion_mnist
    (_, _), (x, y) = fashion_mnist.load_data()
    return x.reshape(-1, 28, 28, 1), y


def _cifar10():
    from tensorflow.keras.datasets import cifar10
    (_, _), (x, y) = cifar10.load_data()
    return x, y.ravel()


def _svhn():
    """Labels as in raw_data._load_svhn: LabelBinarizer over 1..10, then argmax (= y - 1)."""
    from scipy.io import loadmat
    from sklearn.preprocessing import LabelBinarizer
    path = RAW_DATA_DIR / "svhn" / "test_32x32.mat"
    if not path.exists():
        raise MissingDataError(f"SVHN test file not found: {path}")
    raw = loadmat(str(path))
    x = np.moveaxis(np.array(raw["X"]), -1, 0).reshape(-1, 32, 32, 3)
    y = np.argmax(np.asarray(LabelBinarizer().fit_transform(raw["y"])), axis=1)
    return x, y


def _fruit360():
    """fruit_x_test_origin.npy holds 0..255 values as float32 (100x100 images, no resampling)."""
    fruit_dir = RAW_DATA_DIR / "fruit360"
    x_path, y_path = fruit_dir / "fruit_x_test_origin.npy", fruit_dir / "fruit_y_test.npy"
    if not x_path.exists() or not y_path.exists():
        raise MissingDataError(f"Fruit360 test arrays missing in {fruit_dir} (build_fruit_npy.py)")
    x = np.load(x_path)
    if not np.array_equal(x, np.round(x)) or x.min() < 0 or x.max() > 255:
        raise ValueError("Fruit360 test images are not integer 0..255 values; refusing to cast to uint8")
    return x.astype(np.uint8), np.load(y_path).astype(int).ravel()


def _tinyimagenet():
    """val/ images in val_annotations.txt order; class index = sorted train class folders
    (ImageFolder's class_to_idx), as in the torch backend."""
    from PIL import Image
    root = RAW_DATA_DIR / "tiny-imagenet-200"
    if not root.exists():
        raise MissingDataError(f"TinyImageNet not found at {root}")
    class_to_idx = {c: i for i, c in enumerate(sorted(
        d for d in os.listdir(root / "train") if (root / "train" / d).is_dir()))}
    images, labels = [], []
    with open(root / "val" / "val_annotations.txt") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            with Image.open(root / "val" / "images" / tokens[0]) as img:
                images.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
            labels.append(class_to_idx[tokens[1]])
    return np.stack(images), np.array(labels)


RAW_TEST_LOADERS = {
    "mnist": _mnist,
    "fashion_mnist": _fashion,
    "cifar10": _cifar10,
    "svhn": _svhn,
    "fruit360": _fruit360,
    "tinyimagenet": _tinyimagenet,
}


def load_raw_test(dataset):
    """Return (x uint8 (N, H, W, C), y int (N,)) for a dataset."""
    x, y = RAW_TEST_LOADERS[dataset]()
    return np.ascontiguousarray(x, dtype=np.uint8), np.asarray(y).astype(int).ravel()

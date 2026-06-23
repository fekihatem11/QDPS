"""Test/validation split for the RQ4 retraining experiment.

Each subject has a vendored split file (``RETRAIN_SPLITS/{subject}.pkl``) holding
**T** — the list of global test indices used as the selection pool (the SETS paper
divides each test set into T and V). The evaluation set **V** is the complement of
T over the test set; selected inputs (drawn from T) never overlap V, so the
augmented inputs cannot leak into evaluation.
"""
import pickle

from qdps.io.paths import RETRAIN_SPLITS

# Split .pkl filenames per subject_key (exact on-disk casing — do not f-string).
SPLIT_FILE_MAP = {
    "mnist_LeNet1": "mnist_LeNet1.pkl",
    "mnist_LeNet5": "mnist_LeNet5.pkl",
    "Fashion_mnist_LeNet4": "Fashion_mnist_LeNet4.pkl",
    "cifar10_12Conv": "cifar10_12Conv.pkl",
    "cifar10_ResNet20": "cifar10_ResNet20.pkl",
    "SVHN_LeNet5": "SVHN_LeNet5.pkl",
    "Fruit360_ResNet50": "Fruit360_ResNet50.pkl",
    "TinyImageNet_ResNet101": "TinyImagenet_ResNet101.pkl",
}


def load_T(subject_key):
    """Return the selection pool T (list of global test indices)."""
    if subject_key not in SPLIT_FILE_MAP:
        raise KeyError(f"Unknown subject_key: {subject_key}")
    path = RETRAIN_SPLITS / SPLIT_FILE_MAP[subject_key]
    if not path.exists():
        raise FileNotFoundError(
            f"Retrain split not found: {path}. Vendor it from "
            f"SETS/Input_data/Retrain/{SPLIT_FILE_MAP[subject_key]}"
        )
    with open(path, "rb") as f:
        T = pickle.load(f)
    return list(T)


def make_V(T, n_test):
    """Evaluation set V = test indices not in T."""
    return sorted(set(range(n_test)) - set(T))


def selection_pool(T, valid_index):
    """Candidate pool for selection: T restricted to the loader's valid indices."""
    return sorted(set(T) & set(valid_index))

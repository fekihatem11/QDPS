"""The DeepGD retraining protocol (Aghababaeyan et al., TOSEM 2024, RQ2), as applied here.

Decisions (2026-10-27):
- 8 SETS subjects; selection pool = the whole test set (valid indices), no T/V split.
- Budget k = 300. Methods: QDPS and SETS. 5 retraining runs per condition.
- QDPS/SETS retraining = original training set + the k inputs; fresh pretrained model,
  per-subject training settings from config_for() below (2026-10-28, step 2).
- Each retrained model is evaluated on the original test set (T) and on the combined
  generated set (G = the 5 transformed sets together).

Conditions (each retrained N_RUNS times with seeds 0..N_RUNS-1):
    QDPS, SETS   training set + the k selected test inputs
    FULL_T       training set + the whole test set       (max improvement on T)
    FULL_G       training set + the whole generated set  (max improvement on G)
The max-improvement conditions follow the DeepGD paper ("we retrain the original model by
adding all generated inputs to the training dataset"), which is what produced its Table 7.
DeepGD's notebook (Retraining.ipynb cells 34, 39-40) instead fine-tunes on the inputs only
(+ the test set for G); that definition gave effectiveness up to ~200% (MNIST: 6x fewer
training steps than QDPS/SETS) and was dropped on 2026-10-28. Its runs are kept in the
result files as FULL_T_inputs_only / FULL_G_inputs_only.

Metrics:
    improvement on T or G  = acc(retrained) - acc(original)
    optimization effectiveness on T = imp_T(method) / imp_T(FULL_T)
    optimization effectiveness on G = imp_G(method) / imp_G(FULL_G)
"""
from dataclasses import replace

import numpy as np

from qdps.io.paths import GENERATED_DATA_DIR
from qdps.retrain.retrain_deepgd_version.augment_params import OPS
from qdps.retrain.retrain_step import config_for as base_config_for

K = 300
METHODS = ["QDPS", "SETS"]
MAX_CONDITIONS = ["FULL_T", "FULL_G"]
CONDITIONS = METHODS + MAX_CONDITIONS
N_RUNS = 5

# FULL_G inputs: the generated inputs only (paper). The notebook also adds the test set.
FULL_G_INCLUDES_TEST = False

# Max-improvement conditions add their inputs to the training set (paper).
MAX_CONDITIONS_WITH_TRAIN_SET = True

# Retraining settings per subject (decided 2026-10-28). DeepGD published only MNIST
# LeNet5's (Adadelta 0.005, 30 epochs, batch 50), used for both MNIST subjects; with it
# SVHN's maxima were negative and the others far below DeepGD's Table 7. The others were
# chosen by tune_settings.py / tune_report.py on a held-out split, never looking at QDPS
# or SETS (results: RESULTS_DIR/retrain_deepgd/tuning/TUNING.md). Fruit-360's last
# layer is linear, so its loss takes logits. TinyImageNet keeps SETS's setting.
# Epochs, batch size and monitoring split stay as in qdps/retrain/retrain_step.py.
TUNED_SETTINGS = {
    "Fashion_mnist_LeNet4": dict(optimizer="adam", lr=0.0001),
    "cifar10_12Conv": dict(optimizer="adadelta", lr=0.05),
    "cifar10_ResNet20": dict(optimizer="adadelta", lr=0.05),
    "SVHN_LeNet5": dict(optimizer="adam", lr=0.0001),
    "Fruit360_ResNet50": dict(optimizer="adam", lr=0.0001, from_logits=True),
}


def config_for(subject):
    return replace(base_config_for(subject), **TUNED_SETTINGS.get(subject, {}))


# subject -> dataset of its generated test set
SUBJECT_DATASET = {
    "mnist_LeNet1": "mnist",
    "mnist_LeNet5": "mnist",
    "Fashion_mnist_LeNet4": "fashion_mnist",
    "cifar10_12Conv": "cifar10",
    "cifar10_ResNet20": "cifar10",
    "SVHN_LeNet5": "svhn",
    "Fruit360_ResNet50": "fruit360",
    "TinyImageNet_ResNet101": "tinyimagenet",
}


def load_generated_combined(subject):
    """The combined generated set G: (x uint8 (5N, H, W, C), y int (5N,)), ops in OPS order."""
    d = GENERATED_DATA_DIR / SUBJECT_DATASET[subject]
    xs = [np.load(d / f"{op}_test_x.npy") for op in OPS]
    ys = [np.load(d / f"{op}_test_y.npy") for op in OPS]
    return np.concatenate(xs), np.concatenate(ys).astype(int)


def model_input(subject, x_uint8):
    """Keras-model preprocessing of uint8 images: [-0.5, 0.5] for the SETS subjects,
    raw 0..255 float for Fruit-360 (as in retrain_fruit.py)."""
    if subject == "Fruit360_ResNet50":
        return x_uint8.astype("float32")
    return x_uint8.astype("float32") / 255.0 - 0.5

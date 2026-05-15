"""Train one LeNet5 instance on MNIST for a given seed.

Output: ./instances/seed_<n>/model.h5
        ./instances/seed_<n>/history.json
        ./instances/seed_<n>/meta.json

Usage:
    python train.py --seed 0
    python train.py --seed 0 --epochs 30

Architecture is an exact reconstruction of
    SETS/Input_data/Pretrained_model/model_mnist_LeNet5.h5
The original uses linear Conv2D / Dense layers followed by explicit Activation
layers (functionally equivalent to ``activation='relu'`` inside the layer);
this structure is preserved so layer count and naming match the original.

Variance source: the seed controls
    (1) which 80% subset of the 60k MNIST training pool this instance sees
        (split-seed protocol),
    (2) weight initialization, batch shuffling, dropout, augmentation noise.
The test set is fixed across all instances.
"""
from __future__ import annotations

# Seed env vars must be set BEFORE tensorflow is imported.
import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

# ---------------------------------------------------------------------------
# Training recipe: behavioral-match strategy (train-to-target)
# ---------------------------------------------------------------------------
# The SETS pretrained model (model_mnist_LeNet5.h5) reaches test_acc = 0.8785
# under the canonical [-0.5, 0.5] pre-processing. That model is a deliberately
# undertrained variant whose training recipe is NOT published in any public
# repo (ATS, DeepGD, SETS, FATS). So instead of guessing the recipe, we match
# the operational accuracy: train and pick the epoch whose val_accuracy is
# closest to the SETS target (the callback handles weight restoration).
MAX_EPOCHS = 50          # hard cap; closest-to-target normally fires in 1-3 epochs
BATCH_SIZE = 128
OPTIMIZER = "sgd"
LEARNING_RATE = 0.005    # gives epoch-level granularity around the target
SGD_MOMENTUM = 0.0
SGD_NESTEROV = False
TRAIN_POOL_RATIO = 0.8
VALIDATION_RATIO = 0.1

SETS_REFERENCE_ACCURACY = 0.8785
TARGET_VAL_ACCURACY = SETS_REFERENCE_ACCURACY
# +/- 3 pp envelope to accommodate the val/test generalization gap
# (small val set, epoch-granular stopping). See mnist_LeNet1/train.py.
ACCURACY_GATE = 0.03


def seed_everything(seed: int) -> None:
    """Seed every RNG source. Call before importing tensorflow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass


def load_mnist_split(seed: int):
    from tensorflow.keras.datasets import mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    rng = np.random.default_rng(seed)
    pool_size = int(round(TRAIN_POOL_RATIO * len(X_train_full)))
    pool_idx = np.sort(rng.choice(len(X_train_full), size=pool_size, replace=False))

    val_rng = np.random.default_rng(seed + 10_000)
    perm = val_rng.permutation(pool_size)
    n_val = int(round(VALIDATION_RATIO * pool_size))
    train_idx = pool_idx[perm[n_val:]]
    val_idx = pool_idx[perm[:n_val]]

    def prep(X, y):
        # Pre-processing must match the SETS pretrained model: pixels in
        # [-0.5, 0.5], NOT [0, 1]. Verified by loading
        # SETS/Input_data/Pretrained_model/model_mnist_LeNet5.h5 and checking
        # its predictions against SETS/Input_data/Fault_clusters/mnist_LeNet5/
        # mis_index_test.npy -- only [-0.5, 0.5] reproduces the recorded 1215
        # misclassifications (acc = 0.8785). Cf. CLIP_MIN/CLIP_MAX = -0.5/0.5
        # in SETS/Source_code/cluster.py.
        X = X.astype("float32")[..., None] / 255.0 - 0.5
        y_oh = np.eye(10, dtype="float32")[y.astype("int64")]
        return X, y_oh

    X_tr, y_tr = prep(X_train_full[train_idx], y_train_full[train_idx])
    X_va, y_va = prep(X_train_full[val_idx], y_train_full[val_idx])
    X_te, y_te = prep(X_test, y_test)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), pool_idx


def build_lenet5():
    """LeNet5 -- exact reconstruction of SETS' model_mnist_LeNet5.h5."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(name="LeNet5")
    model.add(layers.Input(shape=(28, 28, 1)))

    model.add(layers.Conv2D(6, (5, 5), padding="same",
                            kernel_initializer="glorot_uniform"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (5, 5), padding="same",
                            kernel_initializer="glorot_uniform"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(120, kernel_initializer="glorot_uniform"))
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(84, kernel_initializer="glorot_uniform"))
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(10, kernel_initializer="glorot_uniform"))
    model.add(layers.Activation("softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.legacy.SGD(
            learning_rate=LEARNING_RATE, momentum=SGD_MOMENTUM, nesterov=SGD_NESTEROV
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_stop_at_target_callback(target: float):
    """Restore weights from the epoch whose val_accuracy is closest to `target`,
    and stop once val_accuracy has crossed target and the gap is increasing.
    See mnist_LeNet1/train.py for the full rationale.
    """
    import tensorflow as tf

    class StopAtTargetAccuracy(tf.keras.callbacks.Callback):
        def __init__(self, target: float):
            super().__init__()
            self.target = target
            self.best_epoch = -1
            self.best_val_acc = None
            self.best_gap = float("inf")
            self.best_weights = None
            self.crossed_target = False
            self.stopped_epoch = -1

        def on_epoch_end(self, epoch, logs=None):
            val = (logs or {}).get("val_accuracy")
            if val is None:
                return
            gap = abs(val - self.target)
            if gap < self.best_gap:
                self.best_gap = gap
                self.best_epoch = epoch
                self.best_val_acc = float(val)
                self.best_weights = [w.copy() for w in self.model.get_weights()]
            if val >= self.target:
                self.crossed_target = True
            if self.crossed_target and gap > self.best_gap:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f"[StopAtTargetAccuracy] crossed target {self.target:.4f}; "
                      f"closest epoch={self.best_epoch + 1} val_acc={self.best_val_acc:.4f} "
                      f"(gap={self.best_gap:.4f}); stopping at epoch {epoch + 1}.")

        def on_train_end(self, logs=None):
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                print(f"[StopAtTargetAccuracy] Restored weights from epoch "
                      f"{self.best_epoch + 1} (val_acc={self.best_val_acc:.4f}).")

    return StopAtTargetAccuracy(target)


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description="Train one LeNet5/MNIST instance.")
    p.add_argument("--seed", type=int, required=True, help="Split seed (controls everything stochastic)")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                   help=f"Hard cap on epochs (default {MAX_EPOCHS}); EarlyStopping usually triggers earlier")
    args = p.parse_args()

    seed_everything(args.seed)

    import tensorflow as tf
    print(f"[mnist_LeNet5] seed={args.seed} tf={tf.__version__} max_epochs={args.max_epochs}")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), pool_idx = load_mnist_split(args.seed)
    model = build_lenet5()
    model.summary(print_fn=print)
    print(f"[mnist_LeNet5] train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    stop_cb = make_stop_at_target_callback(TARGET_VAL_ACCURACY)

    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        batch_size=BATCH_SIZE, epochs=args.max_epochs,
        callbacks=[stop_cb],
        verbose=2, shuffle=True,
    )
    train_seconds = time.time() - t0
    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)

    delta = test_acc - SETS_REFERENCE_ACCURACY
    gate_pass = abs(delta) <= ACCURACY_GATE
    n_epochs_run = len(history.history["loss"])
    hit_target = bool(stop_cb.crossed_target)
    print(f"[mnist_LeNet5] done in {train_seconds:.1f}s  epochs_run={n_epochs_run}  "
          f"crossed_target={hit_target}  best_epoch={stop_cb.best_epoch + 1}  "
          f"best_val_acc={stop_cb.best_val_acc:.4f}  "
          f"test_acc={test_acc:.4f}  delta_vs_SETS={delta:+.4f}  "
          f"gate={'PASS' if gate_pass else 'FAIL'} ({ACCURACY_GATE:+.0%})")

    out = HERE / "instances" / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.h5")
    with (out / "history.json").open("w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f, indent=2)
    with (out / "meta.json").open("w") as f:
        json.dump({
            "subject": "mnist_LeNet5",
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "epochs_run": n_epochs_run,
            "stopping_strategy": "closest_to_target_val_accuracy",
            "target_val_accuracy": TARGET_VAL_ACCURACY,
            "crossed_target": hit_target,
            "best_epoch": int(stop_cb.best_epoch) + 1,
            "best_val_accuracy": stop_cb.best_val_acc,
            "best_gap": float(stop_cb.best_gap),
            "batch_size": BATCH_SIZE,
            "optimizer": OPTIMIZER,
            "learning_rate": LEARNING_RATE,
            "sgd_momentum": SGD_MOMENTUM,
            "sgd_nesterov": SGD_NESTEROV,
            "train_pool_ratio": TRAIN_POOL_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te)),
            "pool_indices_sha256_prefix": hashlib.sha256(pool_idx.tobytes()).hexdigest()[:16],
            "train_seconds": train_seconds,
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "sets_reference_accuracy": SETS_REFERENCE_ACCURACY,
            "accuracy_gate_tolerance": ACCURACY_GATE,
            "delta_vs_sets_reference": float(delta),
            "gate_passed": bool(gate_pass),
            "git_commit": git_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        }, f, indent=2)
    print(f"[mnist_LeNet5] saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-instance prediction script.

For a trained robustness instance (subject + seed), this script:
  1. Reloads the model from instances/seed_<n>/model.h5
  2. Regenerates the 48 000-image training pool deterministically from the
     seed using train.py's exact recipe, then verifies the regeneration
     against the pool_indices_sha256_prefix logged in meta.json.
  3. Runs softmax inference on the full MNIST train (60 000) and test
     (10 000) sets.
  4. Identifies misclassifications, filtering the train side to (pool n mis)
     since the cluster.py protocol consumes only the misclassifications
     this instance was actually trained on.
  5. Saves artifacts cluster.py needs.

Output (in instances/seed_<n>/predictions/):

  output_probability_train.npy   (60000, 10)  float32
  output_probability_test.npy    (10000, 10)  float32
  mis_index_train.npy            int64        # GLOBAL indices into 60k, filtered to pool n mis
  mis_index_test.npy             int64        # GLOBAL indices into 10k (all test mis)
  meta.json                                    # provenance + sanity checks

Why predict on the full 60k instead of just the 48k pool: it's cheap
(~5 s on CPU) and keeps the saved softmax arrays seed-independent in
SHAPE, so any seed's predictions can be indexed by ANY mis_index_*
array without re-aligning. The pool n mis filter is encoded in
mis_index_train.npy itself.

Why GLOBAL indices (not pool-local): matches SETS's convention --
mis_index_test.npy under SETS/Input_data/Fault_clusters/<subject>/
holds indices into the full test set. cluster.py will do
    features_train_60k[mis_index_train]
    features_test_10k[mis_index_test]
with no pool reconstruction needed.

Usage:
    python qdps/robustness/predict.py --subject mnist_LeNet1 --seed 0
"""
from __future__ import annotations

# Seed env vars before importing tensorflow (same as train.py).
import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Must match train.py exactly -- if either of these changes there, the pool
# regeneration here will fail the sha256 check and predict aborts.
TRAIN_POOL_RATIO = 0.8


def regenerate_pool(seed: int, n_total: int = 60000) -> np.ndarray:
    """Bit-identical reproduction of the pool selection in train.py.

    train.py:
        rng = np.random.default_rng(seed)
        pool_size = int(round(TRAIN_POOL_RATIO * len(X_train_full)))
        pool_idx = np.sort(rng.choice(len(X_train_full), size=pool_size,
                                       replace=False))
    """
    rng = np.random.default_rng(seed)
    pool_size = int(round(TRAIN_POOL_RATIO * n_total))
    return np.sort(rng.choice(n_total, size=pool_size, replace=False))


def pool_hash_prefix(pool_idx: np.ndarray, length: int = 16) -> str:
    return hashlib.sha256(pool_idx.tobytes()).hexdigest()[:length]


def preprocess(X: np.ndarray) -> np.ndarray:
    """SETS pretrained model convention: [-0.5, 0.5] with channel dim.

    Matches train.py's `prep` exactly (minus the one-hot encoding of y).
    """
    return X.astype("float32")[..., None] / 255.0 - 0.5


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description="Predict for one robustness instance.")
    p.add_argument("--subject", required=True, choices=["mnist_LeNet1", "mnist_LeNet5"],
                   help="Subject directory under qdps/robustness/")
    p.add_argument("--seed", type=int, required=True,
                   help="Which seed_<n> instance to predict for")
    p.add_argument("--force", action="store_true",
                   help="Re-predict even if outputs already exist")
    args = p.parse_args()

    inst_dir = HERE / args.subject / "instances" / f"seed_{args.seed}"
    if not inst_dir.is_dir():
        print(f"[predict] ERROR: instance dir not found: {inst_dir}", file=sys.stderr)
        return 2

    meta_path = inst_dir / "meta.json"
    model_path = inst_dir / "model.h5"
    if not meta_path.exists() or not model_path.exists():
        print(f"[predict] ERROR: missing meta.json or model.h5 in {inst_dir}", file=sys.stderr)
        return 2

    meta = json.loads(meta_path.read_text())
    if meta["seed"] != args.seed:
        print(f"[predict] ERROR: meta seed={meta['seed']} != CLI seed={args.seed}",
              file=sys.stderr)
        return 2

    out_dir = inst_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "prob_train":   out_dir / "output_probability_train.npy",
        "prob_test":    out_dir / "output_probability_test.npy",
        "mis_train":    out_dir / "mis_index_train.npy",
        "mis_test":     out_dir / "mis_index_test.npy",
        "meta":         out_dir / "meta.json",
    }
    if all(p.exists() for p in expected.values()) and not args.force:
        print(f"[predict] {args.subject} seed={args.seed}: outputs exist, "
              f"skipping (use --force to overwrite)")
        return 0

    # Regenerate the pool and verify it matches what train.py logged.
    pool_idx = regenerate_pool(args.seed)
    expected_hash = meta["pool_indices_sha256_prefix"]
    actual_hash = pool_hash_prefix(pool_idx)
    if actual_hash != expected_hash:
        print(f"[predict] ERROR: pool sha256 mismatch -- regenerated={actual_hash} "
              f"meta={expected_hash}. The training recipe may have drifted.",
              file=sys.stderr)
        return 3
    print(f"[predict] {args.subject} seed={args.seed}: pool verified ({actual_hash})")

    # Load the model.
    import tensorflow as tf
    print(f"[predict] tf={tf.__version__} loading model from {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Load MNIST (we always need the full train + test sets).
    from tensorflow.keras.datasets import mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    if len(X_train_full) != 60000 or len(X_test) != 10000:
        print(f"[predict] ERROR: unexpected MNIST shapes train={len(X_train_full)} "
              f"test={len(X_test)}", file=sys.stderr)
        return 4

    Xp_train = preprocess(X_train_full)  # (60000, 28, 28, 1) float32
    Xp_test = preprocess(X_test)         # (10000, 28, 28, 1) float32

    # Inference.
    t0 = time.time()
    prob_train = model.predict(Xp_train, batch_size=512, verbose=0).astype("float32")
    prob_test = model.predict(Xp_test, batch_size=512, verbose=0).astype("float32")
    predict_seconds = time.time() - t0

    # Misclassifications.
    pred_train = np.argmax(prob_train, axis=1).astype("int64")
    pred_test = np.argmax(prob_test, axis=1).astype("int64")
    y_train_int = y_train_full.astype("int64")
    y_test_int = y_test.astype("int64")

    # Train-side mis: filter to pool n mis. pool_idx is sorted; the result
    # therefore stays sorted, matching SETS's mis_index_test.npy convention.
    mis_train_global = pool_idx[pred_train[pool_idx] != y_train_int[pool_idx]]
    mis_test_global = np.where(pred_test != y_test_int)[0].astype("int64")

    # Sanity: cross-check test accuracy against the value train.py logged.
    test_acc = float((pred_test == y_test_int).mean())
    test_acc_meta = float(meta["test_accuracy"])
    test_acc_delta = abs(test_acc - test_acc_meta)
    if test_acc_delta > 1e-4:
        print(f"[predict] WARNING: test_acc {test_acc:.6f} != meta.test_accuracy "
              f"{test_acc_meta:.6f}  (delta {test_acc_delta:.2e})", file=sys.stderr)

    # Pool-level accuracy is informational.
    pool_acc = float((pred_train[pool_idx] == y_train_int[pool_idx]).mean())

    print(f"[predict] {args.subject} seed={args.seed}  "
          f"pool: {len(mis_train_global)}/{len(pool_idx)} mis (acc {pool_acc:.4f})  "
          f"test: {len(mis_test_global)}/{len(y_test_int)} mis (acc {test_acc:.4f})  "
          f"({predict_seconds:.1f}s)")

    # Save.
    np.save(expected["prob_train"], prob_train)
    np.save(expected["prob_test"], prob_test)
    np.save(expected["mis_train"], mis_train_global)
    np.save(expected["mis_test"], mis_test_global)

    pred_meta = {
        "subject": args.subject,
        "seed": args.seed,
        "n_train_full": int(len(X_train_full)),
        "n_test": int(len(X_test)),
        "pool_size": int(len(pool_idx)),
        "pool_indices_sha256_prefix": actual_hash,
        "n_mis_in_pool": int(len(mis_train_global)),
        "n_mis_in_test": int(len(mis_test_global)),
        "pool_accuracy": pool_acc,
        "test_accuracy": test_acc,
        "test_accuracy_meta": test_acc_meta,
        "test_accuracy_delta_vs_meta": test_acc_delta,
        "mis_index_convention": "GLOBAL into full arrays (60000-train and 10000-test); "
                                 "train side filtered to pool n mis",
        "preprocessing": "(x/255) - 0.5 with channel dim",
        "predict_seconds": predict_seconds,
        "model_h5": str(model_path.relative_to(REPO_ROOT)),
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    expected["meta"].write_text(json.dumps(pred_meta, indent=2))

    print(f"[predict] wrote {len(expected)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

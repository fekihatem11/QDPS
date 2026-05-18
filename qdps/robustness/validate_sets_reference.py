"""DIAGNOSTIC ONLY -- not part of the published pipeline.

Validates the QDPS clustering pipeline against the SETS pretrained model
(`model_mnist_LeNet1.h5`, trained on the FULL 60k MNIST training set).

SETS's published `fault_clusters/mnist_LeNet1/cluster_results.npy` has
**137 clusters** on the union of (full train mis) + (full test mis). If
we run our cluster.py sweep on the SETS model's predictions, the winning
config should land near 137 -- if so, our pipeline is faithful and any
deviation we see on our retrained instances comes from the model, not
from the methodology.

This script:
  1. Loads the SETS pretrained model (path via --model-path)
  2. Predicts on full MNIST train (60k) + test (10k) with [-0.5, 0.5] preproc
  3. Computes ALL train mis + ALL test mis (no pool filter -- the SETS
     model saw the whole train set)
  4. Writes outputs in the same format predict.py writes, under
     `instances/mnist_LeNet1/seed_<N>/predictions/` (default N=999 so it
     doesn't collide with our retrained seeds 0..4)
  5. Meta.json gets a `VALIDATION_NOTE` flag so it can never be confused
     with a real instance.

After this script:
  sbatch --export=ALL,SUBJECT=mnist_LeNet1,SEED=999 \
         qdps/robustness/slurm/cluster/cluster_sweep_array.sh
  sbatch --dependency=afterok:<JID> --export=ALL,SUBJECT=mnist_LeNet1,SEED=999 \
         qdps/robustness/slurm/cluster/aggregate_sweep.sh

Then read sweep_results.json's winner and compare its n_clusters to 137.

Cleanup at end:
  rm -rf $SCRATCH/QDPS/instances/mnist_LeNet1/seed_999
  rm -f  qdps/robustness/validate_sets_reference.py  (optional)
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import argparse
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


def preprocess(X: np.ndarray) -> np.ndarray:
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
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="Path to SETS's model_mnist_LeNet1.h5")
    p.add_argument("--output-seed", type=int, default=999,
                   help="Write outputs under seed_<N>/ so they don't collide "
                        "with real instances (default 999)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs")
    args = p.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[validate] ERROR: model not found: {model_path}", file=sys.stderr)
        return 2

    out_inst = HERE / "mnist_LeNet1" / "instances" / f"seed_{args.output_seed}"
    out_pred = out_inst / "predictions"
    out_pred.mkdir(parents=True, exist_ok=True)
    expected = {
        "prob_train": out_pred / "output_probability_train.npy",
        "prob_test":  out_pred / "output_probability_test.npy",
        "mis_train":  out_pred / "mis_index_train.npy",
        "mis_test":   out_pred / "mis_index_test.npy",
        "meta":       out_pred / "meta.json",
    }
    if all(p.exists() for p in expected.values()) and not args.force:
        print(f"[validate] outputs exist at {out_pred}, skipping (--force to overwrite)")
        return 0

    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    print(f"[validate] tf={tf.__version__}")
    print(f"[validate] loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    (Xtr, ytr), (Xte, yte) = mnist.load_data()
    if len(Xtr) != 60000 or len(Xte) != 10000:
        print(f"[validate] ERROR: unexpected MNIST sizes train={len(Xtr)} test={len(Xte)}",
              file=sys.stderr)
        return 3

    Xp_train = preprocess(Xtr)
    Xp_test = preprocess(Xte)

    t0 = time.time()
    prob_train = model.predict(Xp_train, batch_size=512, verbose=0).astype("float32")
    prob_test = model.predict(Xp_test, batch_size=512, verbose=0).astype("float32")
    predict_seconds = time.time() - t0

    pred_train = np.argmax(prob_train, axis=1).astype("int64")
    pred_test = np.argmax(prob_test, axis=1).astype("int64")
    ytr_i = ytr.astype("int64")
    yte_i = yte.astype("int64")

    # ALL train mis + ALL test mis (no pool filter)
    mis_train_global = np.where(pred_train != ytr_i)[0].astype("int64")
    mis_test_global  = np.where(pred_test  != yte_i)[0].astype("int64")

    train_acc = float((pred_train == ytr_i).mean())
    test_acc  = float((pred_test  == yte_i).mean())

    # SETS reference numbers (for sanity-check printing only):
    #   test_accuracy 0.8458, n_mis_test 1542, n_mis_total 11296
    #   so n_mis_train = 11296 - 1542 = 9754
    expected_test  = 1542
    expected_train = 9754
    print(f"[validate] train_acc={train_acc:.4f}  n_mis_train={len(mis_train_global)} "
          f"(SETS=9754, delta={len(mis_train_global)-expected_train:+d})")
    print(f"[validate] test_acc ={test_acc:.4f}  n_mis_test={len(mis_test_global)} "
          f"(SETS=1542, delta={len(mis_test_global)-expected_test:+d})")
    print(f"[validate] n_mis_total={len(mis_train_global)+len(mis_test_global)} "
          f"(SETS=11296)  predict={predict_seconds:.1f}s")

    np.save(expected["prob_train"], prob_train)
    np.save(expected["prob_test"],  prob_test)
    np.save(expected["mis_train"],  mis_train_global)
    np.save(expected["mis_test"],   mis_test_global)

    meta = {
        "subject": "mnist_LeNet1",
        "seed": args.output_seed,
        "n_train_full": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "pool_size": int(len(Xtr)),          # SETS pretrained saw full train
        "pool_indices_sha256_prefix": "FULL_TRAIN_NO_POOL",
        "n_mis_in_pool": int(len(mis_train_global)),  # field name kept for predict-meta compat
        "n_mis_in_test": int(len(mis_test_global)),
        "pool_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_accuracy_meta": 0.8458,
        "test_accuracy_delta_vs_meta": test_acc - 0.8458,
        "mis_index_convention": "ALL train/test misclassifications (no pool filter)",
        "preprocessing": "(x/255) - 0.5 with channel dim",
        "predict_seconds": predict_seconds,
        "model_h5": str(model_path),
        "VALIDATION_NOTE": (
            "DIAGNOSTIC ONLY -- not a real robustness instance. Loaded from "
            "SETS's pretrained model_mnist_LeNet1.h5, which was trained on "
            "the FULL 60k MNIST train set. Used only to validate whether our "
            "cluster.py sweep reproduces SETS's published 137-cluster count. "
            "Delete the parent seed_<N>/ directory after the diagnostic completes."
        ),
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    expected["meta"].write_text(json.dumps(meta, indent=2))
    print(f"[validate] wrote {len(expected)} files to {out_pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

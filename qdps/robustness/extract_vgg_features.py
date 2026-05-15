"""Extract VGG16 (block5_conv3) features for a dataset's train and test sets.

This is a one-time, dataset-level script. The features it produces are
identical for every retrained instance of the same subject -- VGG16 has
ImageNet weights (frozen) and runs in inference mode, so the output is a
deterministic function of input pixels only.

Output:  qdps/robustness/features_for_clustering/{dataset}/features_{split}_raw.npy
                                                            features_{split}_meta.json

Pre-processing pipeline (faithful to SETS/Source_code/feature.py
`vgg16_features_GD`, which itself reuses DeepGD's implementation):

    raw uint8 (28x28 grayscale)
       -> RGB stack along axis=-1                              (28, 28, 3)
       -> PIL resize 28->48                                    (48, 48, 3)
       -> astype float32
       -> scale to [-0.5, 0.5]:  x = (x / 255.0) - 0.5
       -> VGG16 (ImageNet, include_top=False).block5_conv3     (3, 3, 512)
       -> reshape                                              (4608,)

The output is the **raw** VGG output (no per-column min-max normalization).
SETS's saved `features_test_*.npy` files are the *normalized* X_scf variant,
used by the QDPS / SETS selection kernel. For SETS's clustering pipeline
(`SETS/Source_code/cluster.py`) the raw features are used -- this script
matches that convention.

Usage:
    python qdps/robustness/extract_vgg_features.py --dataset mnist
"""
from __future__ import annotations

# Seed env vars before importing tensorflow (deterministic VGG inference is
# already deterministic, but we set them for consistency with train.py)
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
FEATURES_DIR = HERE / "features_for_clustering"

VGG_INPUT_SIZE = (48, 48)
VGG_LAYER = "block5_conv3"
VGG_BATCH = 128


def preprocess_for_vgg(X: np.ndarray) -> np.ndarray:
    """Apply the SETS pre-processing chain to images.

    X must be raw uint8 with shape (N, H, W) (grayscale) or (N, H, W, C).
    Returns float32 array with shape (N, 48, 48, 3) in [-0.5, 0.5].
    """
    from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

    if X.ndim == 3:
        # grayscale -> 3 channels by stacking
        X = np.stack([X] * 3, axis=-1)
    elif X.ndim == 4 and X.shape[-1] == 1:
        X = np.repeat(X, 3, axis=-1)

    # PIL resize to 48x48 per-image (scale=False keeps uint8 values)
    resized = np.empty((len(X), VGG_INPUT_SIZE[0], VGG_INPUT_SIZE[1], 3), dtype=np.uint8)
    for i, im in enumerate(X):
        resized[i] = img_to_array(
            array_to_img(im.astype("uint8"), scale=False).resize(VGG_INPUT_SIZE)
        ).astype("uint8")

    out = resized.astype("float32") / 255.0 - 0.5
    return out


def build_vgg_extractor():
    """VGG16 (ImageNet, no top), output at block5_conv3."""
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Model

    inp = layers.Input(shape=(VGG_INPUT_SIZE[0], VGG_INPUT_SIZE[1], 3))
    base = VGG16(weights="imagenet", input_tensor=inp, include_top=False)
    feat_layer = base.get_layer(VGG_LAYER)
    return Model(inputs=base.input, outputs=feat_layer.output)


def extract(X: np.ndarray, extractor) -> np.ndarray:
    """Run the extractor and reshape to (N, 4608) float32."""
    Xp = preprocess_for_vgg(X)
    raw = extractor.predict(Xp, batch_size=VGG_BATCH, verbose=1)
    n = raw.shape[0]
    return raw.reshape(n, -1).astype("float32")


def load_dataset(dataset: str):
    if dataset == "mnist":
        from tensorflow.keras.datasets import mnist
        (Xtr, ytr), (Xte, yte) = mnist.load_data()
        return {"train": (Xtr, ytr), "test": (Xte, yte)}
    raise NotImplementedError(f"Dataset {dataset!r} not yet wired up.")


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description="Extract VGG16 features for a dataset.")
    p.add_argument("--dataset", required=True, choices=["mnist"],
                   help="Dataset name (only 'mnist' supported in this commit)")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if output files already exist")
    args = p.parse_args()

    dataset_dir = FEATURES_DIR / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    print(f"[extract] dataset={args.dataset} tf={tf.__version__}")
    print(f"[extract] output dir: {dataset_dir}")

    splits = load_dataset(args.dataset)
    extractor = build_vgg_extractor()

    for split_name, (X, y) in splits.items():
        out_path = dataset_dir / f"features_{split_name}_raw.npy"
        meta_path = dataset_dir / f"features_{split_name}_meta.json"
        if out_path.exists() and not args.force:
            print(f"[extract] {out_path} exists, skipping (use --force to overwrite)")
            continue

        print(f"[extract] {split_name}: {X.shape} -> extracting features...")
        t0 = time.time()
        feats = extract(X, extractor)
        elapsed = time.time() - t0
        print(f"[extract] {split_name}: features={feats.shape}  min={feats.min():.4f}  "
              f"max={feats.max():.4f}  mean={feats.mean():.4f}  ({elapsed:.1f}s)")

        np.save(out_path, feats)
        with meta_path.open("w") as f:
            json.dump({
                "dataset": args.dataset,
                "split": split_name,
                "n_samples": int(len(X)),
                "n_features": int(feats.shape[1]),
                "vgg_layer": VGG_LAYER,
                "vgg_input_size": list(VGG_INPUT_SIZE),
                "preprocessing": "rgb_stack -> resize 48x48 -> (x/255)-0.5",
                "normalization": "raw VGG output (no per-column min-max)",
                "extraction_seconds": elapsed,
                "feature_min": float(feats.min()),
                "feature_max": float(feats.max()),
                "feature_mean": float(feats.mean()),
                "feature_std": float(feats.std()),
                "git_commit": git_commit(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python": sys.version.split()[0],
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            }, f, indent=2)
        print(f"[extract] saved {out_path} + {meta_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

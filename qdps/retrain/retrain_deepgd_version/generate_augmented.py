"""Generate the transformed test sets of the DeepGD retraining protocol.

For each dataset, every test image is transformed once by each of the 5
transformations (rotation, zoom, shear, brightness, blur) with that dataset's
ATS ranges, giving 5 files of the test-set size. Output, in ATS's layout:

    GENERATED_DATA_DIR/<dataset>/<OP>_test_x.npy   uint8, same shape as the test set
    GENERATED_DATA_DIR/<dataset>/<OP>_test_y.npy   int labels (unchanged by the transformation)
    GENERATED_DATA_DIR/<dataset>/meta.json         ranges, seed, sizes, checksums

Image i of every file is a transformation of test image i. Each image has its
own RNG, seeded by (SEED, op index, i), so the files are reproducible and any
single image can be regenerated on its own.

Usage (from the repo root):
    python -m qdps.retrain.retrain_deepgd_version.generate_augmented mnist fashion_mnist cifar10 svhn
    python -m qdps.retrain.retrain_deepgd_version.generate_augmented fruit360 tinyimagenet   # on Narval
"""
import hashlib
import json
import sys
from datetime import datetime

import numpy as np

from qdps.io.paths import GENERATED_DATA_DIR
from qdps.retrain.retrain_deepgd_version.augment_ops import FILL_MODE, ORDER, OPERATORS
from qdps.retrain.retrain_deepgd_version.augment_params import DAU_PARAMS, OP_NAMES, OPS
from qdps.retrain.retrain_deepgd_version.raw_test import load_raw_test

SEED = 2026


def transform_dataset(x, op, params, seed=SEED):
    """Apply transformation `op` once to every image of x (uint8, (N, H, W, C))."""
    op_index = OPS.index(op)
    func = OPERATORS[op]
    out = np.empty_like(x)
    for i in range(len(x)):
        rng = np.random.default_rng([seed, op_index, i])
        out[i] = func(x[i], params[op], rng)
    return out


def _sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def generate(dataset, seed=SEED):
    params = DAU_PARAMS[dataset]
    x, y = load_raw_test(dataset)
    out_dir = GENERATED_DATA_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{dataset}: {len(x)} test images {x.shape[1:]} -> {out_dir}")

    files = {}
    for op in OPS:
        x_op = transform_dataset(x, op, params, seed)
        np.save(out_dir / f"{op}_test_x.npy", x_op)
        np.save(out_dir / f"{op}_test_y.npy", y)
        changed = float(np.mean(np.any(x_op != x, axis=tuple(range(1, x.ndim)))))
        files[op] = {"x_sha256": _sha256(x_op), "changed_fraction": changed}
        print(f"  {op} ({OP_NAMES[op]}): done, {changed:.1%} of images changed")

    meta = {
        "dataset": dataset,
        "created": datetime.now().isoformat(timespec="seconds"),
        "generator": "ATS gen_data (Dop.py) port, qdps/retrain/retrain_deepgd_version/augment_ops.py",
        "ops": OPS,
        "params": {op: params[op] for op in OPS},
        "seed": seed,
        "rng": "np.random.default_rng([seed, OPS.index(op), image_index])",
        "fill_mode": FILL_MODE,
        "interpolation_order": ORDER,
        "n_images": int(len(x)),
        "image_shape": list(x.shape[1:]),
        "source_test_x_sha256": _sha256(x),
        "files": files,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["mnist", "fashion_mnist", "cifar10", "svhn"]
    unknown = [d for d in datasets if d not in DAU_PARAMS]
    if unknown:
        sys.exit(f"Unknown dataset(s): {unknown}. Available: {', '.join(DAU_PARAMS)}")
    for d in datasets:
        generate(d)

"""Validity check of the generated test sets (inspect + accuracy report).

For each dataset:
- a sample grid: N random test images (rows) x [original + the 5 transformations] (columns);
- the accuracy of each original (pre-retraining) model on the original test set
  and on each transformed set.

There is no automatic threshold: the report is reviewed before retraining.

Output: RESULTS_DIR/retrain_deepgd/generated_check/{<dataset>_grid.png, REPORT.md}

Usage (from the repo root):
    python -m qdps.retrain.retrain_deepgd_version.check_augmented mnist fashion_mnist cifar10 svhn
"""
import json
import sys

import numpy as np

from qdps.io.paths import GENERATED_DATA_DIR, RESULTS_DIR
from qdps.retrain.raw_data import load_pretrained_model
from qdps.retrain.retrain_deepgd_version.augment_params import DATASET_SUBJECTS, OP_NAMES, OPS
from qdps.retrain.retrain_deepgd_version.protocol import model_input
from qdps.retrain.retrain_deepgd_version.raw_test import load_raw_test

CHECK_DIR = RESULTS_DIR / "retrain_deepgd" / "generated_check"
N_GRID = 8
GRID_SEED = 0


def load_generated(dataset, op):
    d = GENERATED_DATA_DIR / dataset
    return np.load(d / f"{op}_test_x.npy"), np.load(d / f"{op}_test_y.npy")


def accuracy(model, subject, x, y):
    probs = model.predict(model_input(subject, x), batch_size=256, verbose=0)
    return float(np.mean(np.argmax(probs, axis=1) == y))


def save_grid(dataset, x, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.random.default_rng(GRID_SEED).choice(len(x), N_GRID, replace=False)
    columns = [("original", x)] + [(OP_NAMES[op], load_generated(dataset, op)[0]) for op in OPS]
    fig, axes = plt.subplots(N_GRID, len(columns), figsize=(1.6 * len(columns), 1.6 * N_GRID))
    for c, (name, arr) in enumerate(columns):
        axes[0, c].set_title(name, fontsize=9)
        for r, i in enumerate(idx):
            img = arr[i]
            axes[r, c].imshow(img[..., 0] if img.shape[-1] == 1 else img,
                              cmap="gray" if img.shape[-1] == 1 else None, vmin=0, vmax=255)
            axes[r, c].axis("off")
    fig.suptitle(f"{dataset}: test images {sorted(idx.tolist())}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def check(dataset):
    x, y = load_raw_test(dataset)
    for op in OPS:
        _, y_op = load_generated(dataset, op)
        assert np.array_equal(y_op, y), f"{dataset}/{op}: labels differ from the test set"
    save_grid(dataset, x, CHECK_DIR / f"{dataset}_grid.png")

    rows = {}
    for subject in DATASET_SUBJECTS[dataset]:
        model = load_pretrained_model(subject)
        if isinstance(model, str):
            print(f"  {subject}: torch model, accuracy check runs with the torch backend (skipped here)")
            continue
        rows[subject] = {"original": accuracy(model, subject, x, y)}
        for op in OPS:
            x_op, _ = load_generated(dataset, op)
            rows[subject][op] = accuracy(model, subject, x_op, y)
        print(f"  {subject}: " + ", ".join(f"{k} {v:.3f}" for k, v in rows[subject].items()))
    return rows


def write_report(all_rows):
    lines = ["# Generated test sets: validity check", "",
             "Accuracy of each original (pre-retraining) model on the original test set and on each "
             "transformed set. \"Combined (all 5)\" is the accuracy on the 5 sets together, the generated test set of the DeepGD protocol (equal sizes, so it is the mean of the 5 columns); the per-transformation columns are diagnostics.", "",
             "| Subject | Original | " + " | ".join(OP_NAMES[op].capitalize() for op in OPS) + " | Combined (all 5) |",
             "|---|---|" + "---|" * (len(OPS) + 1)]
    for dataset, rows in all_rows.items():
        for subject, acc in rows.items():
            gen = [acc[op] for op in OPS]
            lines.append(f"| {subject} | {acc['original']:.1%} | " + " | ".join(f"{a:.1%}" for a in gen)
                         + f" | {np.mean(gen):.1%} |")
    lines += ["", "Sample grids: " + ", ".join(f"`{d}_grid.png`" for d in all_rows) + "."]
    with open(CHECK_DIR / "REPORT.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(CHECK_DIR / "accuracy.json", "w") as f:
        json.dump(all_rows, f, indent=2)


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["mnist", "fashion_mnist", "cifar10", "svhn"]
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    previous = {}
    if (CHECK_DIR / "accuracy.json").exists():
        previous = json.load(open(CHECK_DIR / "accuracy.json"))
    for d in datasets:
        print(d)
        previous[d] = check(d)
    write_report(previous)
    print(f"Report: {CHECK_DIR / 'REPORT.md'}")

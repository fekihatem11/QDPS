"""Precompute the RQ4 (SETS-protocol) selections, so retraining nodes need no VGG features.

For each subject: pool = T (SETS's split) restricted to the valid (non-noisy) indices,
exactly as SETS/Source_code/exp_4.py builds it; QDPS = qdps.select (final version:
ALPHA = 3, product kernel), SETS = qdps.baselines.sets.sets_select (identical output
to SETS/Source_code/SETS.py::sets(..., "maxp", "gd", a=3), checked on all 8 subjects at
k = 500 on 2026-10-28). Writes qdps/retrain/selections_k<k>.json (tracked in git).

Run with the timing venv, as the SETS FDR baseline (SETS's Fruit-360 output depends on
the BLAS; QDPS's does not):
    .venv-timing/bin/python -m qdps.retrain.precompute_selections 500
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from qdps.experiments.run_single_subject import SUBJECT_MAP
from qdps.io.loader import load_subject
from qdps.qdps import ALPHA
from qdps.retrain.selectors import SELECTORS
from qdps.retrain.splits import load_T, selection_pool

SUBJECTS = ["mnist_LeNet1", "mnist_LeNet5", "Fashion_mnist_LeNet4", "cifar10_12Conv",
            "cifar10_ResNet20", "SVHN_LeNet5", "Fruit360_ResNet50", "TinyImageNet_ResNet101"]


def selections_file(k):
    return Path(__file__).parent / f"selections_k{k}.json"


def load_precomputed(subject, k, method, pool):
    """The precomputed selection, or None if there is no file / entry. The stored pool
    must equal the runner's pool."""
    path = selections_file(k)
    if not path.exists():
        return None
    entry = json.load(open(path))["subjects"].get(subject)
    if entry is None or method not in entry:
        return None
    assert entry["pool_size"] == len(pool), f"{path.name}: pool of {subject} differs"
    return entry[method]


def main(k):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    out = {"k": k, "qdps_alpha": ALPHA, "qdps_kernel": "product", "commit": commit,
           "numpy": np.__version__, "timestamp": datetime.now().isoformat(timespec="seconds"),
           "subjects": {}}
    for subject in SUBJECTS:
        dn, mn = SUBJECT_MAP[subject]
        sel = load_subject(dn, mn)
        assert sel["features"] is not None, f"no VGG features for {subject}"
        pool = selection_pool(load_T(subject), sel["index"])
        entry: dict = {"pool_size": len(pool)}
        for method in ("QDPS", "SETS"):
            chosen = SELECTORS[method](k, pool, sel["features"], sel["output_probability"])
            entry[method] = [int(i) for i in chosen]
        entry["overlap"] = len(set(entry["QDPS"]) & set(entry["SETS"]))
        out["subjects"][subject] = entry
        print(f"{subject}: pool {len(pool)}, QDPS/SETS overlap {entry['overlap']}/{k}", flush=True)
    json.dump(out, open(selections_file(k), "w"))
    print(f"Written {selections_file(k)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)

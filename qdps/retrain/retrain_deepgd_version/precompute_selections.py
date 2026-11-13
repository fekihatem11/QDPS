"""Precompute the k = 300 selections of QDPS and SETS for the retraining experiment.

Selection is cheap and deterministic, so it is done once here (on the Mac) instead of
on the cluster: the clusters then need neither the VGG features nor the fault clusters.
Same inputs and call as the FDR experiments (qdps/experiments/run_single_subject.py):
the loader's valid test indices, in the loader's order (order breaks uncertainty ties).
Run in .venv-timing, the environment of the official SETS baseline
(results/sets_baseline/); QDPS selections are identical in both venvs.

Each selection's FDR is recorded so it can be checked against the official tables.

Output: qdps/retrain/retrain_deepgd_version/selections_k300.json

Usage (repo root):
    .venv-timing/bin/python -m qdps.retrain.retrain_deepgd_version.precompute_selections
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from qdps.experiments.run_single_subject import SUBJECT_MAP
from qdps.io.loader import compute_fdr, load_subject
from qdps.retrain.retrain_deepgd_version.protocol import K, METHODS, SUBJECT_DATASET
from qdps.retrain.selectors import SELECTORS

SELECTIONS_FILE = Path(__file__).with_name(f"selections_k{K}.json")


def main():
    out = {"k": K, "env": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "machine": platform.platform()},
           "subjects": {}}
    for subject in SUBJECT_DATASET:
        dn, mn = SUBJECT_MAP[subject]
        sel = load_subject(dn, mn)
        entry = {"pool_size": len(sel["index"])}
        for method in METHODS:
            chosen = SELECTORS[method](K, sel["index"], sel["features"], sel["output_probability"])
            chosen = [int(i) for i in chosen]
            assert len(chosen) == K and len(set(chosen)) == K, f"{subject}/{method}: bad selection"
            fdr, _ = compute_fdr(chosen, sel["mis_index"], sel["cluster_labels"], K, sel["total_faults"])
            entry[method] = {"selected": chosen, "fdr": float(fdr)}
            print(f"{subject:24s} {method}: FDR {fdr:.4f}")
        out["subjects"][subject] = entry
    SELECTIONS_FILE.write_text(json.dumps(out, indent=1))
    print(f"Written {SELECTIONS_FILE}")


if __name__ == "__main__":
    main()

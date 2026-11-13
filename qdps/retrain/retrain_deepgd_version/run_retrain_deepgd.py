"""Retraining experiment, DeepGD protocol (see protocol.py for the decisions and metrics).

For one subject: measure the original model on T (test set) and G (combined generated
set), then for each requested condition retrain N_RUNS times and measure again.
Results are merged into RESULTS_DIR/retrain_deepgd/subjects/<subject>.json, so the
conditions of a subject can run as separate jobs.

Usage (repo root):
    python -m qdps.retrain.retrain_deepgd_version.run_retrain_deepgd <subject> [conditions] [n_runs]
    e.g. ... run_retrain_deepgd mnist_LeNet1                    # QDPS,SETS,FULL_T,FULL_G, 5 runs
         ... run_retrain_deepgd Fruit360_ResNet50 FULL_G 5
"""
import fcntl
import json
import sys
from datetime import datetime

import numpy as np

from qdps.experiments.run_single_subject import SUBJECT_MAP
from qdps.io.paths import RESULTS_DIR
from qdps.retrain.raw_data import load_pretrained_model, load_raw_subject
from qdps.retrain.retrain_deepgd_version import backends
from qdps.retrain.retrain_deepgd_version.protocol import (
    CONDITIONS, FULL_G_INCLUDES_TEST, K, MAX_CONDITIONS_WITH_TRAIN_SET, METHODS, N_RUNS,
    config_for, load_generated_combined, model_input,
)
from qdps.retrain.retrain_deepgd_version.precompute_selections import SELECTIONS_FILE

SUBJECTS_DIR = RESULTS_DIR / "retrain_deepgd" / "subjects"


def select_inputs(subject, method):
    """The k test indices chosen by `method`, precomputed by precompute_selections.py
    (same selections as the official FDR results, so no features are needed here)."""
    data = json.load(open(SELECTIONS_FILE))
    assert data["k"] == K, f"{SELECTIONS_FILE.name} holds k={data['k']}, protocol k={K}"
    entry = data["subjects"][subject]
    return entry[method]["selected"], entry["pool_size"]


def _test_labels(raw):
    if raw.framework == "torch":
        return np.asarray(raw.val_dataset.classidx)
    return raw.y_test_int


def run_condition(subject, condition, raw, g_x, g_y, cfg, n_runs):
    """Retrain n_runs times under one condition; returns the condition's result dict."""
    model_loader = lambda: load_pretrained_model(subject)  # noqa: E731
    with_train = condition in METHODS or MAX_CONDITIONS_WITH_TRAIN_SET
    info: dict = {"n_extra": None, "with_train_set": with_train, "retrain_config": vars(cfg)}

    if condition in METHODS:
        selected, pool_size = select_inputs(subject, condition)
        info.update(selected=selected, pool_size=pool_size, n_extra=len(selected))
    elif condition == "FULL_T":
        selected = list(range(len(_test_labels(raw))))
        info["n_extra"] = len(selected)
    elif condition == "FULL_G":
        selected = None
        info["n_extra"] = len(g_y) + (len(_test_labels(raw)) if FULL_G_INCLUDES_TEST else 0)
        info["full_g_includes_test"] = FULL_G_INCLUDES_TEST
    else:
        raise ValueError(f"Unknown condition {condition}")

    runs = []
    for seed in range(n_runs):
        if raw.framework == "torch":
            from torch.utils.data import ConcatDataset, Subset
            if selected is not None:
                extra = Subset(raw.val_dataset, selected)
            else:
                extra = backends.ArrayDataset(g_x, g_y)
                if FULL_G_INCLUDES_TEST:
                    extra = ConcatDataset([extra, raw.val_dataset])
            acc_t, acc_g = backends.torch_retrain(raw, model_loader, extra, cfg, seed, g_x, g_y,
                                                  with_train=with_train)
        else:
            if selected is not None:
                extra_x, extra_y = raw.x_test[selected], raw.y_test_int[selected]
            else:
                extra_x, extra_y = model_input(subject, g_x), g_y
                if FULL_G_INCLUDES_TEST:
                    extra_x = np.concatenate([extra_x, raw.x_test])
                    extra_y = np.concatenate([extra_y, raw.y_test_int])
            acc_t, acc_g = backends.keras_retrain(raw, subject, model_loader, extra_x, extra_y,
                                                  cfg, seed, g_x, g_y, with_train=with_train)
            del extra_x
        runs.append({"seed": seed, "acc_T": acc_t, "acc_G": acc_g})
        print(f"  {condition} run {seed + 1}/{n_runs}: acc(T)={acc_t:.4f} acc(G)={acc_g:.4f}", flush=True)
    info["runs"] = runs
    return info


def _store_path(subject):
    return SUBJECTS_DIR / f"{subject}.json"


def save_merged(subject, update):
    """Merge `update` into the subject's store (other conditions' results are kept)."""
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _store_path(subject)
    with open(path.with_suffix(".lock"), "w") as lock:  # parallel jobs of one subject
        fcntl.flock(lock, fcntl.LOCK_EX)
        store = json.load(open(path)) if path.exists() else {"subject": subject, "conditions": {}}
        store["original"] = update["original"]
        store["meta"] = update["meta"]
        store["conditions"].update(update["conditions"])
        with open(path, "w") as f:
            json.dump(store, f, indent=2)
    return path


def run_subject(subject, conditions=CONDITIONS, n_runs=N_RUNS):
    cfg = config_for(subject)
    raw = load_raw_subject(subject)
    g_x, g_y = load_generated_combined(subject)
    n_test = len(_test_labels(raw))
    assert len(g_y) == 5 * n_test and np.array_equal(g_y[:n_test], _test_labels(raw)), \
        "generated labels do not match the test set"

    if raw.framework == "torch":
        acc_t, acc_g = backends.torch_original(raw, lambda: load_pretrained_model(subject), g_x, g_y)
    else:
        acc_t, acc_g = backends.keras_original(raw, subject, lambda: load_pretrained_model(subject), g_x, g_y)
    print(f"{subject}: |T|={n_test} |G|={len(g_y)} original acc(T)={acc_t:.4f} acc(G)={acc_g:.4f}")

    meta = {"protocol": "DeepGD RQ2", "k": K, "n_runs": n_runs,
            "retrain_config": vars(cfg), "full_g_includes_test": FULL_G_INCLUDES_TEST,
            "max_conditions_with_train_set": MAX_CONDITIONS_WITH_TRAIN_SET,
            "timestamp": datetime.now().isoformat(timespec="seconds")}
    for condition in conditions:
        result = run_condition(subject, condition, raw, g_x, g_y, cfg, n_runs)
        path = save_merged(subject, {"original": {"acc_T": acc_t, "acc_G": acc_g},
                                     "meta": meta, "conditions": {condition: result}})
        print(f"  saved {condition} -> {path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in SUBJECT_MAP:
        sys.exit(__doc__)
    conds = args[1].split(",") if len(args) > 1 else CONDITIONS
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        sys.exit(f"Unknown condition(s) {unknown}; choose from {CONDITIONS}")
    run_subject(args[0], conds, int(args[2]) if len(args) > 2 else N_RUNS)

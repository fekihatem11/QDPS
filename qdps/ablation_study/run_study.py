"""
Runner for the QDPS ablation studies.

Each study lives in its own folder `qdps/ablation_study/<study>/` with a
`method.py` (exposing METHOD_NAME, CONFIG, select) and a `results/` folder.
Results of every run are written to `<study>/results/<timestamp>/`.

Usage (from the repo root):
    python -m qdps.ablation_study.run_study <study> [n_runs] [subjects] [budgets]
    python -m qdps.ablation_study.run_study s1_parameter_free 1
    python -m qdps.ablation_study.run_study s1_parameter_free 1 mnist_LeNet1 100,300
    python -m qdps.ablation_study.run_study all 1
"""
import importlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from qdps.io.loader import DATA_MODEL_PAIRS, load_subject, compute_fdr
from qdps.io.paths import FAULT_CLUSTERS
from qdps.ablation_study.plot_run import plot_run, write_tables

STUDY_ROOT = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [100, 300, 500]


def list_studies():
    return sorted(
        name for name in os.listdir(STUDY_ROOT)
        if os.path.isfile(os.path.join(STUDY_ROOT, name, "method.py"))
    )


def parse_subjects(arg):
    if arg is None or arg == "all":
        return DATA_MODEL_PAIRS
    wanted = arg.split(",")
    pairs = [(d, m) for d, m in DATA_MODEL_PAIRS if f"{d}_{m}" in wanted]
    unknown = set(wanted) - {f"{d}_{m}" for d, m in pairs}
    if unknown:
        raise SystemExit(f"Unknown subject(s): {sorted(unknown)}")
    return pairs


def run_study(study, n_runs=30, subjects=None, budgets=None):
    module = importlib.import_module(f"qdps.ablation_study.{study}.method")
    subjects = subjects or DATA_MODEL_PAIRS
    budgets = budgets or BUDGETS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(STUDY_ROOT, study, "results", timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Study: {study} ({module.METHOD_NAME})")
    print(f"Config: {module.CONFIG}")
    all_results = {}

    for data_name, model_name in subjects:
        subject_key = f"{data_name}_{model_name}"
        print(f"\n{'='*60}\nSubject: {subject_key}\n{'='*60}")
        subject = load_subject(data_name, model_name, FAULT_CLUSTERS)

        subject_results = {}
        for budget in budgets:
            runs = []
            for _ in range(n_runs):
                start = time.time()
                selected = module.select(
                    size=budget,
                    index=subject["index"],
                    features=subject["features"],
                    output_probability=subject["output_probability"],
                )
                elapsed = time.time() - start
                fdr, faults = compute_fdr(
                    selected, subject["mis_index"], subject["cluster_labels"],
                    budget, subject["total_faults"],
                )
                runs.append({"fdr": fdr, "faults_found": faults, "time": elapsed,
                             "subset_size": len(selected)})

            fdrs = [r["fdr"] for r in runs]
            times = [r["time"] for r in runs]
            subject_results[budget] = {
                "runs": runs,
                "mean_fdr": float(np.mean(fdrs)),
                "std_fdr": float(np.std(fdrs)),
                "min_fdr": float(np.min(fdrs)),
                "max_fdr": float(np.max(fdrs)),
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
            }
            print(f"  k={budget}: FDR={np.mean(fdrs):.4f} +/- {np.std(fdrs):.4f}, "
                  f"time={np.mean(times):.3f}s")
        all_results[subject_key] = subject_results

    meta = {
        "method": module.METHOD_NAME,
        "study": study,
        "config": module.CONFIG,
        "timestamp": timestamp,
        "n_runs": n_runs,
        "budgets": budgets,
        "subjects": [f"{d}_{m}" for d, m in subjects],
    }
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2)

    with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
        f.write(f"Study: {study}\nMethod: {module.METHOD_NAME}\n")
        f.write(f"Config: {module.CONFIG}\nRuns per config: {n_runs}\n{'='*80}\n\n")
        f.write(f"{'Subject':<30} {'k':>5} {'Mean FDR':>10} {'Std FDR':>10} {'Mean Time':>12}\n")
        f.write(f"{'-'*80}\n")
        for subject_key, subject_results in all_results.items():
            for budget in budgets:
                sr = subject_results[budget]
                f.write(f"{subject_key:<30} {budget:>5} {sr['mean_fdr']:>10.4f} "
                        f"{sr['std_fdr']:>10.4f} {sr['mean_time']:>12.4f}s\n")

    print(f"\nResults saved to: {exp_dir}")
    print(f"Plot saved to: {plot_run(exp_dir)}")
    print(f"Tables saved to: {write_tables(exp_dir)}")
    return all_results, exp_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: run_study <study|all> [n_runs] [subjects] [budgets]\n"
                         f"Studies: {', '.join(list_studies())}")
    study_arg = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    subjects = parse_subjects(sys.argv[3] if len(sys.argv) > 3 else None)
    budgets = [int(b) for b in sys.argv[4].split(",")] if len(sys.argv) > 4 else None

    studies = list_studies() if study_arg == "all" else [study_arg]
    for study in studies:
        run_study(study, n_runs=n_runs, subjects=subjects, budgets=budgets)

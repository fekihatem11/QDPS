"""
Experiment runner for QDPS.
Runs QDPS on all subjects, saves results with full provenance.
"""
import sys
import os
import json
import time
import numpy as np
from datetime import datetime

from data_loader import DATA_MODEL_PAIRS, load_subject, compute_fdr
from qdps import select as qdps_select, METHOD_NAME

BUDGETS = [100, 300, 500]
BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "fault_clusters")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_single(subject, budget):
    """Run QDPS on a single subject with a single budget."""
    start = time.time()
    selected = qdps_select(
        size=budget,
        index=subject["index"],
        features=subject["features"],
        output_probability=subject["output_probability"],
    )
    elapsed = time.time() - start

    fdr, faults_found = compute_fdr(
        selected, subject["mis_index"], subject["cluster_labels"],
        budget, subject["total_faults"]
    )
    return {
        "fdr": fdr,
        "faults_found": faults_found,
        "time": elapsed,
        "subset_size": len(selected),
    }


def run_experiment(n_runs=30, subjects=None, budgets=None):
    """Run full experiment: QDPS x subjects x budgets x n_runs."""
    if subjects is None:
        subjects = DATA_MODEL_PAIRS
    if budgets is None:
        budgets = BUDGETS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{METHOD_NAME}_{timestamp}"
    exp_dir = os.path.join(RESULTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    all_results = {}

    for data_name, model_name in subjects:
        subject_key = f"{data_name}_{model_name}"
        print(f"\n{'='*60}")
        print(f"Subject: {subject_key}")
        print(f"{'='*60}")

        try:
            subject = load_subject(data_name, model_name, BASE_DATA_PATH)
        except Exception as e:
            print(f"  SKIP: Failed to load data: {e}")
            continue

        subject_results = {}
        for budget in budgets:
            print(f"\n  Budget k={budget}:")
            run_results = []

            for run_idx in range(n_runs):
                result = run_single(subject, budget)
                run_results.append(result)
                if run_idx == 0 or (run_idx + 1) % 10 == 0:
                    print(f"    Run {run_idx+1}/{n_runs}: FDR={result['fdr']:.4f}, "
                          f"Faults={result['faults_found']}, Time={result['time']:.4f}s")

            fdrs = [r["fdr"] for r in run_results]
            times = [r["time"] for r in run_results]
            subject_results[budget] = {
                "runs": run_results,
                "mean_fdr": float(np.mean(fdrs)),
                "std_fdr": float(np.std(fdrs)),
                "min_fdr": float(np.min(fdrs)),
                "max_fdr": float(np.max(fdrs)),
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
            }
            print(f"  => Mean FDR: {np.mean(fdrs):.4f} +/- {np.std(fdrs):.4f}, "
                  f"Mean Time: {np.mean(times):.4f}s")

        all_results[subject_key] = subject_results

    # Save results
    meta = {
        "method": METHOD_NAME,
        "timestamp": timestamp,
        "n_runs": n_runs,
        "budgets": budgets,
        "subjects": [f"{d}_{m}" for d, m in subjects],
    }

    results_file = os.path.join(exp_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2)

    summary_file = os.path.join(exp_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Experiment: {exp_id}\n")
        f.write(f"Method: {METHOD_NAME}\n")
        f.write(f"Runs per config: {n_runs}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Subject':<30} {'k':>5} {'Mean FDR':>10} {'Std FDR':>10} {'Mean Time':>12}\n")
        f.write(f"{'-'*80}\n")
        for subject_key, subject_results in all_results.items():
            for budget in budgets:
                if budget in subject_results:
                    sr = subject_results[budget]
                    f.write(f"{subject_key:<30} {budget:>5} {sr['mean_fdr']:>10.4f} "
                            f"{sr['std_fdr']:>10.4f} {sr['mean_time']:>12.4f}s\n")

    print(f"\nResults saved to: {exp_dir}")
    return all_results, exp_dir


if __name__ == "__main__":
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"Running {METHOD_NAME}")
    print(f"Runs per config: {n_runs}")
    results, exp_dir = run_experiment(n_runs=n_runs)

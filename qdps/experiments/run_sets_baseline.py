"""
Run SETS baseline on all subjects (30 runs each) and store results.
This ensures SETS and QDPS timings are measured on the same machine.

Results saved to: qdps/sets_results/
"""
import os
import sys
import time
import json
import numpy as np

from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA, SETS_RESULTS_DIR as RESULTS_DIR
from qdps.baselines.sets import sets_select

N_RUNS = 30
BUDGETS = [100, 300, 500]


# ---- Runner ----

def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    for dn, mn in DATA_MODEL_PAIRS:
        key = f"{dn}_{mn}"
        print(f"\n{'='*70}")
        print(f"  Subject: {key}")
        print(f"{'='*70}")

        try:
            subject = load_subject(dn, mn, BASE_DATA)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        subject_results = {}

        for k in BUDGETS:
            print(f"\n  k={k}:")
            fdrs = []
            times = []

            for i in range(N_RUNS):
                start = time.time()
                selected = sets_select(
                    k, subject['index'], subject['features'],
                    subject['output_probability']
                )
                elapsed = time.time() - start

                fdr, faults = compute_fdr(
                    selected, subject['mis_index'], subject['cluster_labels'],
                    k, subject['total_faults']
                )
                fdrs.append(fdr)
                times.append(elapsed)

                if i == 0 or (i + 1) % 10 == 0:
                    print(f"    Run {i+1:>2}/{N_RUNS}: FDR={fdr:.4f}  "
                          f"Faults={faults}  Time={elapsed:.4f}s")

            subject_results[k] = {
                'fdrs': fdrs,
                'times': times,
                'mean_fdr': float(np.mean(fdrs)),
                'std_fdr': float(np.std(fdrs)),
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
            }

            print(f"    => Mean FDR: {np.mean(fdrs):.4f} +/- {np.std(fdrs):.4f}  "
                  f"Mean Time: {np.mean(times):.4f}s")

        all_results[key] = subject_results

    # Save results
    results_file = os.path.join(RESULTS_DIR, "sets_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save summary
    summary_file = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"SETS Baseline Results ({N_RUNS} runs per configuration)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Subject':<30} {'k':>5} {'Mean FDR':>10} {'Std FDR':>10} "
                f"{'Mean Time':>12} {'Std Time':>12}\n")
        f.write(f"{'─'*80}\n")
        for key, sr in all_results.items():
            for k in BUDGETS:
                if k in sr:
                    r = sr[k]
                    f.write(f"{key:<30} {k:>5} {r['mean_fdr']:>10.4f} "
                            f"{r['std_fdr']:>10.4f} {r['mean_time']:>12.4f}s "
                            f"{r['std_time']:>12.4f}s\n")

    print(f"\n{'='*70}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_all()

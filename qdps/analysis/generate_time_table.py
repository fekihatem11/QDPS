"""
Generate the execution-time comparison table: QDPS vs SETS (plain text).
Runs QDPS 30 times per configuration to get mean timing.
Uses SETS timing from results/sets_baseline/sets_results.json.
"""
import os
import json
import time
import numpy as np
from qdps import select as qdps_select
from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA, SETS_RESULTS_DIR, FDR_RESULTS_DIR

SETS_RESULTS = os.path.join(SETS_RESULTS_DIR, "sets_results.json")
BUDGETS = [100, 300, 500]
N_RUNS = 30

SUBJECTS_DISPLAY = [
    ("mnist", "LeNet1", "MNIST", "LeNet1"),
    ("mnist", "LeNet5", "", "LeNet5"),
    ("Fashion_mnist", "LeNet4", "Fashion", "LeNet4"),
    ("cifar10", "12Conv", "CIFAR-10", "12Conv"),
    ("cifar10", "ResNet20", "", "ResNet20"),
    ("SVHN", "LeNet5", "SVHN", "LeNet5"),
    ("Fruit360", "ResNet50", "Fruit-360", "ResNet50"),
    ("TinyImageNet", "ResNet101", "TinyImageNet", "ResNet101"),
]


def generate():
    # Load SETS timings
    with open(SETS_RESULTS, 'r') as f:
        sets_data = json.load(f)

    # Run QDPS 30 times per config for timing
    qdps_times = {}
    for dn, mn, _, _ in SUBJECTS_DISPLAY:
        key = f"{dn}_{mn}"
        print(f"Timing QDPS on {key}...")
        subject = load_subject(dn, mn, BASE_DATA)
        qdps_times[key] = {}

        for k in BUDGETS:
            times = []
            for i in range(N_RUNS):
                start = time.time()
                _ = qdps_select(k, subject['index'], subject['features'],
                                subject['output_probability'])
                elapsed = time.time() - start
                times.append(elapsed)

            mean_t = np.mean(times)
            qdps_times[key][k] = mean_t
            sets_t = sets_data[key][str(k)]['mean_time']
            print(f"  k={k}: QDPS={mean_t:.4f}s  SETS={sets_t:.4f}s  "
                  f"speedup={sets_t/mean_t:.1f}x")

    # Build plain-text table
    lines = []
    lines.append("QDPS vs SETS — selection time in seconds (* = faster)")
    lines.append("=" * 78)
    lines.append(f"{'Subject':<26}{'k=100':^16}{'k=300':^16}{'k=500':^16}")
    lines.append(f"{'':<26}{'QDPS':>7}{'SETS':>8}{'QDPS':>8}{'SETS':>8}{'QDPS':>8}{'SETS':>8}")
    lines.append("-" * 78)

    qdps_faster_count = 0
    total = 0

    for dn, mn, _, _ in SUBJECTS_DISPLAY:
        key = f"{dn}_{mn}"
        cols = []
        for k in BUDGETS:
            qt = qdps_times[key][k]
            st = sets_data[key][str(k)]['mean_time']
            total += 1
            if qt < st:
                qdps_faster_count += 1
                cols.append(f"{qt:.2f}*")
                cols.append(f"{st:.2f} ")
            else:
                cols.append(f"{qt:.2f} ")
                cols.append(f"{st:.2f}*")
        lines.append(f"{key:<26}" + "".join(f"{c:>8}" for c in cols))

    lines.append("-" * 78)
    lines.append(f"QDPS faster in {qdps_faster_count}/{total} configurations")
    table_str = "\n".join(lines) + "\n"

    # Save
    os.makedirs(os.path.join(FDR_RESULTS_DIR, "tables"), exist_ok=True)
    output_path = os.path.join(FDR_RESULTS_DIR, "tables", "table_time_qdps_vs_sets.txt")
    with open(output_path, "w") as f:
        f.write(table_str)

    print(f"\n{table_str}")
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    generate()

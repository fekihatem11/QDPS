"""
Generate LaTeX execution time comparison table: QDPS vs SETS.
Runs QDPS 30 times per configuration to get mean timing.
Uses SETS timing from sets_results/sets_results.json.
"""
import os
import json
import time
import numpy as np
from data_loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps import select as qdps_select

BASE_DATA = os.path.join(os.path.dirname(__file__), "fault_clusters")
SETS_RESULTS = os.path.join(os.path.dirname(__file__), "sets_results", "sets_results.json")
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

    # Generate LaTeX
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Execution Time Costs (in Seconds) of QDPS and SETS Approaches}")
    latex.append(r"\label{tab:time_qdps_vs_sets}")
    latex.append(r"\begin{tabular}{ll|cc|cc|cc}")
    latex.append(r"\hline")
    latex.append(r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{DNN Model}} "
                 r"& \multicolumn{2}{c|}{\textbf{k = 100}} "
                 r"& \multicolumn{2}{c|}{\textbf{k = 300}} "
                 r"& \multicolumn{2}{c}{\textbf{k = 500}} \\")
    latex.append(r"\cline{3-8}")
    latex.append(r"& & \textbf{QDPS} & \textbf{SETS} "
                 r"& \textbf{QDPS} & \textbf{SETS} "
                 r"& \textbf{QDPS} & \textbf{SETS} \\")
    latex.append(r"\hline")

    qdps_faster_count = 0
    total = 0

    for dn, mn, ds_label, model_label in SUBJECTS_DISPLAY:
        key = f"{dn}_{mn}"
        cols = []

        for k in BUDGETS:
            qt = qdps_times[key][k]
            st = sets_data[key][str(k)]['mean_time']
            total += 1

            # Bold the faster one
            if qt < st:
                qdps_faster_count += 1
                cols.append(f"\\textbf{{{qt:.2f}}}")
                cols.append(f"{st:.2f}")
            else:
                cols.append(f"{qt:.2f}")
                cols.append(f"\\textbf{{{st:.2f}}}")

        row = (f"{ds_label} & {model_label} & {cols[0]} & {cols[1]} "
               f"& {cols[2]} & {cols[3]} & {cols[4]} & {cols[5]} \\\\")
        latex.append(row)

        if (dn == "mnist" and mn == "LeNet5") or (dn == "cifar10" and mn == "ResNet20"):
            latex.append(r"\hline")

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    table_str = "\n".join(latex)

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "table_time_qdps_vs_sets.tex")
    with open(output_path, "w") as f:
        f.write(table_str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"QDPS faster in {qdps_faster_count}/{total} configurations")
    print(f"{'='*60}")
    print(f"\nLaTeX table saved to: {output_path}")
    print(f"\n{table_str}")


if __name__ == "__main__":
    generate()

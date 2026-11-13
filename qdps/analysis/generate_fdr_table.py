"""
Generate the FDR comparison table: QDPS vs SETS (plain text).
'*' marks the better value (beyond the ±0.005 tie band).
"""
import os
import numpy as np
from qdps import select as qdps_select
from qdps.io.loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps.io.paths import FAULT_CLUSTERS as BASE_DATA, FDR_RESULTS_DIR
BUDGETS = [100, 300, 500]

SUBJECTS_DISPLAY = [
    # (data_name, model_name, dataset_label, model_label)
    ("mnist", "LeNet1", "MNIST", "LeNet1"),
    ("mnist", "LeNet5", "", "LeNet5"),
    ("Fashion_mnist", "LeNet4", "Fashion", "LeNet4"),
    ("cifar10", "12Conv", "CIFAR-10", "12Conv"),
    ("cifar10", "ResNet20", "", "ResNet20"),
    ("SVHN", "LeNet5", "SVHN", "LeNet5"),
    ("Fruit360", "ResNet50", "Fruit-360", "ResNet50"),
    ("TinyImageNet", "ResNet101", "TinyImageNet", "ResNet101"),
]

# SETS FDRs from our own run on this machine
SETS_FDRS = {
    'mnist_LeNet1': {100: 0.4600, 300: 0.5474, 500: 0.6277},
    'mnist_LeNet5': {100: 0.4706, 300: 0.7294, 500: 0.7882},
    'Fashion_mnist_LeNet4': {100: 0.3400, 300: 0.5352, 500: 0.6197},
    'cifar10_12Conv': {100: 0.5500, 300: 0.5455, 500: 0.6684},
    'cifar10_ResNet20': {100: 0.5200, 300: 0.5225, 500: 0.6292},
    'SVHN_LeNet5': {100: 0.4600, 300: 0.6351, 500: 0.7230},
    'Fruit360_ResNet50': {100: 0.4100, 300: 0.3133, 500: 0.2860},
    'TinyImageNet_ResNet101': {100: 0.3696, 300: 0.6413, 500: 0.7391},
}


def fmt(value, is_better):
    """Format FDR as percentage; '*' marks the better value."""
    return f"{value:.0%}{'*' if is_better else ' '}"


def generate():
    # Run QDPS once per configuration
    qdps_fdrs = {}
    for dn, mn, _, _ in SUBJECTS_DISPLAY:
        key = f"{dn}_{mn}"
        print(f"Running QDPS on {key}...")
        subject = load_subject(dn, mn, BASE_DATA)
        qdps_fdrs[key] = {}
        for k in BUDGETS:
            selected = qdps_select(k, subject['index'], subject['features'],
                                   subject['output_probability'])
            fdr, _ = compute_fdr(selected, subject['mis_index'],
                                 subject['cluster_labels'], k, subject['total_faults'])
            qdps_fdrs[key][k] = fdr
            print(f"  k={k}: QDPS={fdr:.4f}  SETS={SETS_FDRS[key][k]:.4f}")

    # Build plain-text table
    lines = []
    lines.append("QDPS vs SETS — FDR comparison (* = better by >0.005)")
    lines.append("=" * 78)
    lines.append(f"{'Subject':<26}{'k=100':^16}{'k=300':^16}{'k=500':^16}")
    lines.append(f"{'':<26}{'QDPS':>7}{'SETS':>8}{'QDPS':>8}{'SETS':>8}{'QDPS':>8}{'SETS':>8}")
    lines.append("-" * 78)

    wins = 0
    losses = 0
    ties = 0

    for dn, mn, _, _ in SUBJECTS_DISPLAY:
        key = f"{dn}_{mn}"
        cols = []
        for k in BUDGETS:
            q = qdps_fdrs[key][k]
            s = SETS_FDRS[key][k]
            q_better = q > s + 0.005
            s_better = s > q + 0.005
            cols.append(fmt(q, q_better))
            cols.append(fmt(s, s_better))
            if q_better:
                wins += 1
            elif s_better:
                losses += 1
            else:
                ties += 1
        lines.append(f"{key:<26}" + "".join(f"{c:>8}" for c in cols))

    lines.append("-" * 78)
    lines.append(f"Summary: Wins {wins}  Losses {losses}  Ties {ties}  (out of 24)")
    table_str = "\n".join(lines) + "\n"

    # Save
    os.makedirs(os.path.join(FDR_RESULTS_DIR, "tables"), exist_ok=True)
    output_path = os.path.join(FDR_RESULTS_DIR, "tables", "table_qdps_vs_sets.txt")
    with open(output_path, "w") as f:
        f.write(table_str)

    print(f"\n{table_str}")
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    generate()

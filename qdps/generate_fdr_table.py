"""
Generate LaTeX comparison table: QDPS vs SETS.
Bold the value that is significantly better.
Since both methods are deterministic, bold = strictly higher FDR.
"""
import os
import numpy as np
from data_loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps import select as qdps_select

BASE_DATA = os.path.join(os.path.dirname(__file__), "..", "SETS", "Input_data", "Fault_clusters")
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
    """Format FDR as percentage, bold if better."""
    pct = f"{value:.0%}"
    if is_better:
        return f"\\textbf{{{pct}}}"
    return pct


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

    # Generate LaTeX
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Fault Detection Rates (FDRs) of QDPS and SETS Approaches}")
    latex.append(r"\label{tab:qdps_vs_sets}")
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

    wins = 0
    losses = 0
    ties = 0

    for dn, mn, ds_label, model_label in SUBJECTS_DISPLAY:
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

        row = f"{ds_label} & {model_label} & {cols[0]} & {cols[1]} & {cols[2]} & {cols[3]} & {cols[4]} & {cols[5]} \\\\"
        latex.append(row)

        # Add hline after pairs or single-model datasets
        if mn in ["LeNet5", "ResNet20"] and dn in ["mnist", "cifar10"]:
            latex.append(r"\hline")
        elif dn not in ["mnist", "cifar10"] or (dn == "mnist" and mn == "LeNet5") or (dn == "cifar10" and mn == "ResNet20"):
            pass

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    # Add hlines between dataset groups
    final_latex = []
    for line in latex:
        final_latex.append(line)

    table_str = "\n".join(final_latex)

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "table_qdps_vs_sets.tex")
    with open(output_path, "w") as f:
        f.write(table_str)

    print(f"\n{'='*60}")
    print(f"Summary: QDPS vs SETS")
    print(f"  Wins: {wins}  Losses: {losses}  Ties: {ties}  (out of 24)")
    print(f"{'='*60}")
    print(f"\nLaTeX table saved to: {output_path}")
    print(f"\n{table_str}")


if __name__ == "__main__":
    generate()

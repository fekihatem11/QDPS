"""
Compare QDPS results against published baselines: SETS, DeepGD, and RS.
"""
import warnings
from data_loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps import select as qdps_select
import os

BASE_DATA = os.path.join(os.path.dirname(__file__), "..", "SETS", "Input_data", "Fault_clusters")

# Published SETS FDRs (Paper Table 3)
SETS_PAPER = {
    'mnist_LeNet1': {100: 0.46, 300: 0.55, 500: 0.63},
    'mnist_LeNet5': {100: 0.47, 300: 0.73, 500: 0.79},
    'Fashion_mnist_LeNet4': {100: 0.34, 300: 0.54, 500: 0.62},
    'cifar10_12Conv': {100: 0.55, 300: 0.55, 500: 0.67},
    'cifar10_ResNet20': {100: 0.52, 300: 0.52, 500: 0.63},
    'SVHN_LeNet5': {100: 0.46, 300: 0.64, 500: 0.72},
    'Fruit360_ResNet50': {100: 0.45, 300: 0.32, 500: 0.29},
    'TinyImageNet_ResNet101': {100: 0.37, 300: 0.64, 500: 0.74},
}

# Published DeepGD FDRs (Paper Table 3)
DEEPGD_PAPER = {
    'mnist_LeNet1': {100: 0.34, 300: 0.49, 500: 0.61},
    'mnist_LeNet5': {100: 0.36, 300: 0.60, 500: 0.70},
    'Fashion_mnist_LeNet4': {100: 0.35, 300: 0.48, 500: 0.58},
    'cifar10_12Conv': {100: 0.53, 300: 0.51, 500: 0.66},
    'cifar10_ResNet20': {100: 0.51, 300: 0.54, 500: 0.67},
    'SVHN_LeNet5': {100: 0.41, 300: 0.56, 500: 0.68},
    'Fruit360_ResNet50': {100: 0.37, 300: 0.27, 500: 0.22},
    'TinyImageNet_ResNet101': {100: 0.32, 300: 0.59, 500: 0.71},
}

# RS FDRs (from replication package, mean of 30 runs)
RS_PAPER = {
    'mnist_LeNet1': {100: 0.13, 300: 0.23, 500: 0.35},
    'mnist_LeNet5': {100: 0.11, 300: 0.26, 500: 0.38},
    'Fashion_mnist_LeNet4': {100: 0.10, 300: 0.18, 500: 0.25},
    'cifar10_12Conv': {100: 0.14, 300: 0.22, 500: 0.32},
    'cifar10_ResNet20': {100: 0.13, 300: 0.19, 500: 0.27},
    'SVHN_LeNet5': {100: 0.10, 300: 0.17, 500: 0.26},
    'Fruit360_ResNet50': {100: 0.12, 300: 0.11, 500: 0.10},
    'TinyImageNet_ResNet101': {100: 0.11, 300: 0.28, 500: 0.41},
}


def run_comparison():
    """Run QDPS and compare against all baselines."""
    warnings.filterwarnings('ignore')

    print("=" * 100)
    print("QDPS vs All Baselines - Comprehensive Comparison")
    print("=" * 100)

    headers = ["Subject", "k", "QDPS", "SETS", "DeepGD", "RS", "vs SETS", "vs DeepGD"]
    print(f"{headers[0]:<28} {headers[1]:>4} {headers[2]:>8} {headers[3]:>8} {headers[4]:>8} "
          f"{headers[5]:>8} {headers[6]:>10} {headers[7]:>10}")
    print("-" * 100)

    vs_sets_wins = vs_sets_losses = 0
    vs_dgd_wins = vs_dgd_losses = 0
    total = 0

    for dn, mn in DATA_MODEL_PAIRS:
        s = load_subject(dn, mn, BASE_DATA)
        key = f"{dn}_{mn}"

        for k in [100, 300, 500]:
            sel = qdps_select(k, s['index'], s['features'], s['output_probability'])
            qdps_fdr, _ = compute_fdr(sel, s['mis_index'], s['cluster_labels'], k, s['total_faults'])

            sets_fdr = SETS_PAPER[key][k]
            dgd_fdr = DEEPGD_PAPER[key][k]
            rs_fdr = RS_PAPER[key][k]

            d_sets = qdps_fdr - sets_fdr
            d_dgd = qdps_fdr - dgd_fdr
            total += 1

            if d_sets > 0.005: vs_sets_wins += 1
            elif d_sets < -0.005: vs_sets_losses += 1
            if d_dgd > 0.005: vs_dgd_wins += 1
            elif d_dgd < -0.005: vs_dgd_losses += 1

            s_mark = "+" if d_sets > 0.005 else ("-" if d_sets < -0.005 else "~")
            d_mark = "+" if d_dgd > 0.005 else ("-" if d_dgd < -0.005 else "~")

            print(f"{key:<28} {k:>4} {qdps_fdr:>8.2%} {sets_fdr:>8.0%} {dgd_fdr:>8.0%} "
                  f"{rs_fdr:>8.0%} {d_sets:>+9.1%} {s_mark} {d_dgd:>+9.1%} {d_mark}")

    print("\n" + "=" * 100)
    print(f"QDPS vs SETS:   {vs_sets_wins}W / {vs_sets_losses}L / "
          f"{total-vs_sets_wins-vs_sets_losses}T out of {total}")
    print(f"QDPS vs DeepGD: {vs_dgd_wins}W / {vs_dgd_losses}L / "
          f"{total-vs_dgd_wins-vs_dgd_losses}T out of {total}")


if __name__ == "__main__":
    run_comparison()

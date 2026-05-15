"""
Statistical comparison of QDPS against baselines using Wilcoxon Signed-Rank Test.

Runs QDPS 30 times per configuration and compares against the 30 per-run FDR values
from the SETS replication package (SETS, DeepGD, RS).

The Wilcoxon Signed-Rank Test checks whether there is a statistically significant
difference between two paired samples (significance level = 0.05).
"""
import os
import warnings
import numpy as np
from scipy.stats import wilcoxon

from data_loader import load_subject, compute_fdr, DATA_MODEL_PAIRS
from qdps import select as qdps_select

warnings.filterwarnings('ignore')

BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baseline_results")
BASE_DATA = os.path.join(os.path.dirname(__file__), "fault_clusters")

N_RUNS = 30
BUDGETS = [100, 300, 500]

# Mapping from (data_name, model_name) to the filename prefix used in the replication package
FILENAME_MAP = {
    ("mnist", "LeNet1"): "mnist_LeNet1",
    ("mnist", "LeNet5"): "mnist_LeNet5",
    ("Fashion_mnist", "LeNet4"): "Fashion_mnist_LeNet4",
    ("cifar10", "12Conv"): "cifar10_12Conv",
    ("cifar10", "ResNet20"): "cifar10_ResNet20",
    ("SVHN", "LeNet5"): "SVHN_LeNet5",
    ("Fruit360", "ResNet50"): "fruit360_resnet50",
    ("TinyImageNet", "ResNet101"): "tinyimagenet_resnet101",
}


def load_baseline_fdrs(method, data_name, model_name, budget):
    """Load 30 per-run FDR values from the SETS replication package."""
    prefix = FILENAME_MAP[(data_name, model_name)]
    filepath = os.path.join(BASELINE_DIR, method, f"{prefix}_{budget}.txt")

    fdrs = []
    in_fdr_section = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "FDR List:":
                in_fdr_section = True
                continue
            if in_fdr_section:
                if line == "" or line.endswith("List:"):
                    break
                try:
                    fdrs.append(float(line))
                except ValueError:
                    break

    return np.array(fdrs[:N_RUNS])


def run_qdps_30(subject, budget):
    """Run QDPS 30 times and return FDR values."""
    fdrs = []
    for _ in range(N_RUNS):
        selected = qdps_select(
            size=budget,
            index=subject['index'],
            features=subject['features'],
            output_probability=subject['output_probability'],
        )
        fdr, _ = compute_fdr(
            selected, subject['mis_index'], subject['cluster_labels'],
            budget, subject['total_faults']
        )
        fdrs.append(fdr)
    return np.array(fdrs)


def wilcoxon_test(qdps_fdrs, baseline_fdrs):
    """Perform Wilcoxon Signed-Rank Test.

    Returns (p_value, significant, direction):
        - p_value: the p-value from the test
        - significant: True if p < 0.05
        - direction: '+' if QDPS is better, '-' if baseline is better, '~' if no difference
    """
    diff = qdps_fdrs - baseline_fdrs

    # If all differences are zero, no test needed
    if np.all(diff == 0):
        return 1.0, False, '~'

    # Wilcoxon requires at least one non-zero difference
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 1.0, False, '~'

    try:
        stat, p_value = wilcoxon(qdps_fdrs, baseline_fdrs, alternative='two-sided')
    except ValueError:
        return 1.0, False, '~'

    significant = p_value < 0.05
    mean_diff = np.mean(diff)

    if significant:
        direction = '+' if mean_diff > 0 else '-'
    else:
        direction = '~'

    return p_value, significant, direction


def run_statistical_comparison():
    """Run full statistical comparison: QDPS vs SETS, DeepGD, RS."""

    print("=" * 120)
    print("QDPS Statistical Comparison — Wilcoxon Signed-Rank Test (alpha = 0.05)")
    print(f"30 runs per configuration")
    print("=" * 120)

    for baseline_name in ["SETS", "DeepGD", "RS"]:
        print(f"\n{'─'*120}")
        print(f"  QDPS vs {baseline_name}")
        print(f"{'─'*120}")
        print(f"{'Subject':<28} {'k':>4} {'QDPS Mean':>10} {baseline_name+' Mean':>10} "
              f"{'Delta':>8} {'p-value':>10} {'Sig?':>6} {'Winner':>8}")
        print(f"{'─'*120}")

        wins = 0
        losses = 0
        ties = 0
        total = 0

        for dn, mn in DATA_MODEL_PAIRS:
            key = f"{dn}_{mn}"

            try:
                subject = load_subject(dn, mn, BASE_DATA)
            except Exception as e:
                print(f"  SKIP {key}: {e}")
                continue

            for budget in BUDGETS:
                # Load baseline 30 runs
                try:
                    baseline_fdrs = load_baseline_fdrs(baseline_name, dn, mn, budget)
                except FileNotFoundError:
                    print(f"  {key:<28} {budget:>4}  — baseline file not found")
                    continue

                # Run QDPS 30 times
                qdps_fdrs = run_qdps_30(subject, budget)

                # Wilcoxon test
                p_value, significant, direction = wilcoxon_test(qdps_fdrs, baseline_fdrs)

                total += 1
                if direction == '+':
                    wins += 1
                    winner = "QDPS"
                elif direction == '-':
                    losses += 1
                    winner = baseline_name
                else:
                    ties += 1
                    winner = "—"

                delta = np.mean(qdps_fdrs) - np.mean(baseline_fdrs)
                sig_mark = "YES" if significant else "no"

                print(f"{key:<28} {budget:>4} {np.mean(qdps_fdrs):>10.4f} "
                      f"{np.mean(baseline_fdrs):>10.4f} {delta:>+8.4f} "
                      f"{p_value:>10.4f} {sig_mark:>6} {winner:>8}")

        print(f"\n  Summary vs {baseline_name}: "
              f"{wins}W / {losses}L / {ties}T out of {total} "
              f"(W = QDPS significantly better, L = significantly worse, T = no significant difference)")

    print(f"\n{'='*120}")
    print("Done.")


if __name__ == "__main__":
    run_statistical_comparison()

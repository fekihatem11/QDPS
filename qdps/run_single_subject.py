"""
Run QDPS on a single subject for n runs and display FDR + time results.

Usage:
    python run_single_subject.py <subject> [n_runs] [budgets]

Examples:
    python run_single_subject.py mnist_LeNet1
    python run_single_subject.py cifar10_ResNet20 30
    python run_single_subject.py Fruit360_ResNet50 10 100,300,500

Available subjects:
    mnist_LeNet1, mnist_LeNet5, Fashion_mnist_LeNet4,
    cifar10_12Conv, cifar10_ResNet20, SVHN_LeNet5,
    Fruit360_ResNet50, TinyImageNet_ResNet101
"""
import sys
import time
import numpy as np

from data_loader import load_subject, compute_fdr
from qdps import select as qdps_select
import os

BASE_DATA = os.path.join(os.path.dirname(__file__), "fault_clusters")

BUDGETS = [100, 300, 500]

# Baseline FDRs for comparison
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

# Parse subject string into (data_name, model_name)
SUBJECT_MAP = {
    'mnist_LeNet1': ('mnist', 'LeNet1'),
    'mnist_LeNet5': ('mnist', 'LeNet5'),
    'Fashion_mnist_LeNet4': ('Fashion_mnist', 'LeNet4'),
    'cifar10_12Conv': ('cifar10', '12Conv'),
    'cifar10_ResNet20': ('cifar10', 'ResNet20'),
    'SVHN_LeNet5': ('SVHN', 'LeNet5'),
    'Fruit360_ResNet50': ('Fruit360', 'ResNet50'),
    'TinyImageNet_ResNet101': ('TinyImageNet', 'ResNet101'),
}


def run_subject(subject_key, n_runs=30, budgets=None):
    if budgets is None:
        budgets = BUDGETS

    if subject_key not in SUBJECT_MAP:
        print(f"Unknown subject: {subject_key}")
        print(f"Available: {', '.join(SUBJECT_MAP.keys())}")
        sys.exit(1)

    dn, mn = SUBJECT_MAP[subject_key]
    subject = load_subject(dn, mn, BASE_DATA)

    print(f"{'='*80}")
    print(f"  Subject: {subject_key}")
    print(f"  Runs:    {n_runs}")
    print(f"  Samples: {subject['n_samples']}  Classes: {subject['n_classes']}  "
          f"Faults: {subject['total_faults']}")
    print(f"{'='*80}")

    # Run QDPS
    results = {}
    for k in budgets:
        fdrs = []
        times = []
        for i in range(n_runs):
            start = time.time()
            selected = qdps_select(k, subject['index'], subject['features'],
                                   subject['output_probability'])
            elapsed = time.time() - start
            fdr, faults = compute_fdr(selected, subject['mis_index'],
                                      subject['cluster_labels'], k, subject['total_faults'])
            fdrs.append(fdr)
            times.append(elapsed)

            if n_runs <= 5 or (i + 1) % 10 == 0:
                print(f"  k={k}  Run {i+1:>2}/{n_runs}: FDR={fdr:.4f}  "
                      f"Faults={faults}  Time={elapsed:.4f}s")

        results[k] = {'fdrs': fdrs, 'times': times}

    # Results table
    print(f"\n{'='*80}")
    print(f"  Results: {subject_key} ({n_runs} runs)")
    print(f"{'='*80}")
    print(f"\n{'k':>6} {'QDPS Mean':>10} {'Std':>8} {'Min':>8} {'Max':>8} "
          f"{'Time(s)':>10} {'SETS':>8} {'DeepGD':>8} {'RS':>8}")
    print(f"{'─'*80}")

    for k in budgets:
        fdrs = results[k]['fdrs']
        times = results[k]['times']
        sets_fdr = SETS_PAPER.get(subject_key, {}).get(k, None)
        dgd_fdr = DEEPGD_PAPER.get(subject_key, {}).get(k, None)
        rs_fdr = RS_PAPER.get(subject_key, {}).get(k, None)

        print(f"{k:>6} {np.mean(fdrs):>10.2%} {np.std(fdrs):>8.2%} "
              f"{np.min(fdrs):>8.2%} {np.max(fdrs):>8.2%} "
              f"{np.mean(times):>10.4f} "
              f"{sets_fdr:>8.0%} {dgd_fdr:>8.0%} {rs_fdr:>8.0%}")

    # Delta table
    print(f"\n{'k':>6} {'vs SETS':>10} {'vs DeepGD':>12} {'vs RS':>10}")
    print(f"{'─'*42}")
    for k in budgets:
        qdps_mean = np.mean(results[k]['fdrs'])
        sets_fdr = SETS_PAPER.get(subject_key, {}).get(k, 0)
        dgd_fdr = DEEPGD_PAPER.get(subject_key, {}).get(k, 0)
        rs_fdr = RS_PAPER.get(subject_key, {}).get(k, 0)

        print(f"{k:>6} {qdps_mean - sets_fdr:>+10.2%} "
              f"{qdps_mean - dgd_fdr:>+12.2%} "
              f"{qdps_mean - rs_fdr:>+10.2%}")

    # Per-run detail
    if n_runs <= 10:
        print(f"\n{'─'*80}")
        print(f"  Per-run FDR values:")
        for k in budgets:
            fdrs_str = ", ".join(f"{f:.2%}" for f in results[k]['fdrs'])
            print(f"  k={k}: [{fdrs_str}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    subject_key = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    budgets = [int(b) for b in sys.argv[3].split(',')] if len(sys.argv) > 3 else BUDGETS

    run_subject(subject_key, n_runs, budgets)

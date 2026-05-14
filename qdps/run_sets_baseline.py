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

from data_loader import load_subject, compute_fdr, DATA_MODEL_PAIRS

BASE_DATA = os.path.join(os.path.dirname(__file__), "..", "SETS", "Input_data", "Fault_clusters")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "sets_results")

N_RUNS = 30
BUDGETS = [100, 300, 500]


# ---- SETS implementation (faithful reproduction) ----

def gd(IDs, features):
    selected_features = features[list(IDs)]
    dot_p = np.dot(selected_features, selected_features.T)
    sign, log_det = np.linalg.slogdet(dot_p)
    return log_det


def maxp_score(output_probability):
    return [1 - max(prob) for prob in output_probability]


def sets_select(size, index, features, output_probability, a=3):
    """SETS algorithm — faithful reproduction of the original."""
    un_scores = maxp_score(output_probability)
    sorted_indices = sorted(index, key=lambda i: un_scores[i], reverse=True)

    top_count = max(1, int(a * size))
    if a * size > len(index):
        top_count = len(index)
    filtered = sorted_indices[:top_count]
    chunks = [filtered[i::size] for i in range(size)]

    S = []
    current_gd = 0
    for chunk in chunks:
        if len(chunk) == 0:
            continue

        max_fitness = -float('inf')
        best_idx = -1
        gd_deltas = []
        gd_values = []

        for i in chunk:
            new_gd = gd(S + [i], features)
            gd_values.append(new_gd)
            gd_deltas.append(new_gd - current_gd)

        min_d = min(gd_deltas)
        max_d = max(gd_deltas)
        if max_d - min_d > 0:
            norm_deltas = [(d - min_d) / (max_d - min_d + 0.5) for d in gd_deltas]
        else:
            norm_deltas = [0] * len(gd_deltas)

        for idx_in_chunk, i in enumerate(chunk):
            fitness = un_scores[i] * norm_deltas[idx_in_chunk]
            if fitness > max_fitness:
                max_fitness = fitness
                best_idx = i

        if best_idx != -1:
            S.append(best_idx)
            S_idx = chunk.index(best_idx)
            current_gd = gd_values[S_idx]

    return S


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

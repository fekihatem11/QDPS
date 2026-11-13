"""SETS test-selection algorithm (Wang et al., TOSEM 2025) — faithful reproduction.

Extracted from the FDR runner so both the FDR experiment and the RQ4 retraining
experiment import the selector from one place. `sets_select` has the same
(size, index, features, output_probability) contract as ``qdps.select``.
"""
import numpy as np


def gd(IDs, features):
    selected_features = features[list(IDs)]
    dot_p = np.dot(selected_features, selected_features.T)
    sign, log_det = np.linalg.slogdet(dot_p)
    return log_det


def maxp_score(output_probability):
    return [1 - max(prob) for prob in output_probability]


def sets_select(size, index, features, output_probability, a=3):
    """SETS algorithm — faithful reproduction of the original.

    Returns a list of selected global test indices chosen from ``index``.
    """
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

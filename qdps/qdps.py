"""
QDPS: Quality-Diversity DPP Selection.

A black-box DNN test selection approach that formulates selection as
quality-weighted DPP MAP inference over a visual-behavioural product kernel.

Algorithm (configuration fixed by the ablation study, qdps/ablation_study/):
- Phase 1: Uncertainty u_i = 1 - max(p_i) (MaxP)
- Phase 2: Candidate pool = top ALPHA*k inputs by uncertainty (ALPHA = 3)
- Phase 3: Product kernel K = K_vgg ⊙ K_prob, where K_vgg is the cosine
           similarity of VGG16 features and K_prob the cosine similarity of
           log-probability vectors. Two inputs are similar only if they look
           alike AND the model behaves alike on them.
           L-ensemble: L = diag(u) K diag(u)
- Phase 4: Greedy MAP inference with incremental Cholesky updates, O(N k^2)
- Phase 5: Coverage completion. If the kernel's numerical rank is below k
           (e.g. near-duplicate inputs), the conditional variances reach
           rounding noise before k inputs are chosen. The greedy then stops and
           the remaining slots go to the candidate maximising
           u_j * (1 - max_{s in selected} K(j, s)): an uncertain input far from
           its nearest selected input. Without saturation this phase never runs.

There are no other hyperparameters: no uncertainty reserve, no view weight,
no uncertainty exponent, and no budget-dependent values. The saturation
tolerance is a numerical-precision convention, not a tuned value.
"""
import numpy as np

METHOD_NAME = "QDPS"

# Candidate pool size, as a multiple of the budget k (same for every budget).
ALPHA = 3

EPS = 1e-10

# The greedy stops when the largest conditional variance falls below
# SATURATION_TOL x the first one: below that, values are floating-point noise.
SATURATION_TOL = np.sqrt(np.finfo(np.float64).eps)


def maxp_score(output_probability):
    """MaxP uncertainty metric: 1 - max(p_i)."""
    return 1.0 - np.max(output_probability, axis=1)


def _unit_rows(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
    return x / norms


def select(size, index, features, output_probability):
    """QDPS test selection.

    Args:
        size: budget k (number of test inputs to select)
        index: valid test indices (list or set)
        features: VGG16 features for all test inputs (n_samples x dim), or None
                  to use the behavioural kernel only
        output_probability: model output probabilities (n_samples x n_classes)

    Returns:
        List of selected test indices, in selection order.
    """
    index = list(index)
    un_scores = maxp_score(output_probability)

    # ---- Phase 1-2: uncertainty ranking and candidate pool ----
    sorted_indices = sorted(index, key=lambda i: un_scores[i], reverse=True)
    top_count = min(max(1, int(ALPHA * size)), len(index))
    candidates = sorted_indices[:top_count]
    n_cand = len(candidates)
    cand_un = un_scores[candidates].astype(np.float64)  # float32 softmax -> float64 arithmetic

    # ---- Phase 3: product kernel and L-ensemble ----
    prob_feat = _unit_rows(np.log(output_probability[candidates] + EPS))
    K = prob_feat @ prob_feat.T
    if features is not None:
        vgg_feat = _unit_rows(features[candidates])
        K *= vgg_feat @ vgg_feat.T

    q = cand_un + EPS
    L = np.outer(q, q) * K
    L[np.diag_indices(n_cand)] += EPS  # numerical stability

    # ---- Phase 4: greedy MAP inference with Cholesky updates ----
    n_select = min(size, n_cand)
    d = np.diag(L).copy()  # conditional variances
    c = np.zeros((n_select, n_cand))  # Cholesky factors
    available = np.ones(n_cand, dtype=bool)
    selected_local = []

    tol = None

    for t in range(n_select):
        best_j = int(np.argmax(np.where(available, d, -np.inf)))
        if tol is None:
            tol = SATURATION_TOL * d[best_j]
        if d[best_j] <= tol:
            break  # kernel span exhausted; remaining values are rounding noise

        selected_local.append(best_j)
        available[best_j] = False

        c[t] = (L[best_j] - c[:t, best_j] @ c[:t]) / np.sqrt(d[best_j])
        d = np.maximum(d - c[t] ** 2, 0.0)

    # ---- Phase 5: coverage completion (only if the greedy saturated) ----
    if len(selected_local) < n_select:
        max_sim = K[:, selected_local].max(axis=1) if selected_local else np.zeros(n_cand)
        while len(selected_local) < n_select:
            score = cand_un * (1.0 - np.clip(max_sim, 0.0, 1.0))
            best_j = int(np.argmax(np.where(available, score, -np.inf)))
            selected_local.append(best_j)
            available[best_j] = False
            max_sim = np.maximum(max_sim, K[:, best_j])

    return [candidates[i] for i in selected_local]

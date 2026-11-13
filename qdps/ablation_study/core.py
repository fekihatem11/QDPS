"""
Configurable QDPS engine shared by every ablation study.

Each study's `method.py` calls `qdps_select` with an explicit configuration, so
the only thing that differs between studies is the component being ablated.

`qdps/qdps.py` holds the final method chosen by this study (product kernel,
alpha = 3). This engine keeps every ablated component configurable:
- Every parameter is passed explicitly. The original budget-adaptive values
  are kept below as TUNED_DEFAULTS for the S0 reference.
- `alpha=None` means "no candidate reduction" (pool = all valid test inputs).
- `reserve_ratio=0` disables the reserve *and* the reserved-similarity penalty.
- When the greedy saturates (kernel rank < k), the remaining slots are filled
  by coverage completion, exactly as in `qdps/qdps.py`.
- L is never materialised: rows are computed on demand inside the greedy loop
  (fast greedy MAP, Chen et al. 2018), so memory is O(N*k) instead of O(N^2).
  This is what makes the full pool tractable on SVHN (26k) / Fruit360 (23.6k).
"""
import numpy as np

EPS = 1e-10

# The greedy stops when the largest conditional variance falls below
# SATURATION_TOL x the first one: below that, values are floating-point noise.
SATURATION_TOL = np.sqrt(np.finfo(np.float64).eps)

# Pools up to this size use a dense kernel matrix (see qdps_select).
DENSE_MAX = 5000

# Budget-adaptive values of the original (tuned) QDPS, before the ablation study.
TUNED_DEFAULTS = {
    'small':  {'alpha': 3, 'reserve_ratio': 0.2, 'feature_mix': 0.5, 'quality_power': 2.0},   # k <= 100
    'medium': {'alpha': 4, 'reserve_ratio': 0.1, 'feature_mix': 0.3, 'quality_power': 1.5},   # 100 < k <= 300
    'large':  {'alpha': 3, 'reserve_ratio': 0.1, 'feature_mix': 0.5, 'quality_power': 1.5},   # k > 300
}


def get_tuned_defaults(size):
    """Return the original budget-adaptive hyperparameters for budget `size`."""
    if size <= 100:
        return TUNED_DEFAULTS['small']
    elif size <= 300:
        return TUNED_DEFAULTS['medium']
    return TUNED_DEFAULTS['large']


def maxp_score(output_probability):
    """MaxP uncertainty: 1 - max(p_i)."""
    return 1.0 - np.max(output_probability, axis=1)


def _unit_rows(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
    return x / norms


def qdps_select(size, index, features, output_probability,
                alpha=None, reserve_ratio=0.0, feature_mix=0.5,
                quality_power=1.0, kernel="sum", penalty_strength=0.5):
    """Quality-weighted DPP greedy MAP selection.

    Args:
        size: budget k
        index: valid test indices
        features: VGG16 features (n_samples x dim) or None
        output_probability: softmax outputs (n_samples x n_classes)
        alpha: candidate pool = top alpha*k by uncertainty; None = all inputs
        reserve_ratio: fraction of k taken by pure uncertainty; 0 = no reserve
        feature_mix: weight of K_vgg in the "sum" kernel (1 - mix for K_prob)
        quality_power: exponent p in q_i = u_i^p
        kernel: "sum"     -> K = mix*K_vgg + (1-mix)*K_prob
                "product" -> K = K_vgg * K_prob (elementwise; ignores feature_mix)
        penalty_strength: strength of the reserved-similarity penalty
    """
    index = list(index)
    un_scores = maxp_score(output_probability)
    sorted_indices = sorted(index, key=lambda i: un_scores[i], reverse=True)

    # ---- Phase 1: reserve (optional) ----
    n_reserved = max(1, int(reserve_ratio * size)) if reserve_ratio > 0 else 0
    reserved = sorted_indices[:n_reserved]
    reserved_set = set(reserved)

    # ---- Phase 2: candidate pool (optional reduction) ----
    if alpha is None:
        pool_source = sorted_indices
    else:
        top_count = min(max(1, int(alpha * size)), len(index))
        pool_source = sorted_indices[:top_count]
    candidates = [i for i in pool_source if i not in reserved_set]

    n_dpp = size - n_reserved
    if n_dpp <= 0 or len(candidates) == 0:
        return reserved[:size]
    n_cand = len(candidates)
    cand_un = un_scores[candidates].astype(np.float64)  # float32 softmax -> float64 arithmetic

    # ---- Phase 3: kernel views (unit-normalised, so K_ii = 1 for each view) ----
    prob_feat = _unit_rows(np.log(output_probability[candidates] + EPS))
    use_vgg = features is not None and (kernel == "product" or feature_mix > 0)
    vgg_feat = _unit_rows(features[candidates]) if use_vgg else None

    def kernel_to(prob_rows, vgg_rows):
        """Kernel between every candidate and the given (unit-normalised) rows."""
        k_prob = prob_feat @ prob_rows.T
        if vgg_feat is None:
            return k_prob
        k_vgg = vgg_feat @ vgg_rows.T
        if kernel == "product":
            return k_vgg * k_prob
        return feature_mix * k_vgg + (1.0 - feature_mix) * k_prob

    # Small pools: build the dense kernel with the same arithmetic as qdps/qdps.py,
    # so near-ties (near-duplicate inputs) are broken identically. Large pools
    # (alpha=None on 10k-26k inputs): rows on demand, to keep memory O(N*k).
    K_dense = None
    if n_cand <= DENSE_MAX:
        K_dense = prob_feat @ prob_feat.T
        if vgg_feat is not None:
            if kernel == "product":
                K_dense *= vgg_feat @ vgg_feat.T
            else:
                K_dense = feature_mix * (vgg_feat @ vgg_feat.T) + (1.0 - feature_mix) * K_dense

    def kernel_row(j):
        if K_dense is not None:
            return K_dense[j]
        return kernel_to(prob_feat[j][None, :], None if vgg_feat is None else vgg_feat[j][None, :])[:, 0]

    def kernel_col(j):
        return K_dense[:, j] if K_dense is not None else kernel_row(j)

    q = np.power(cand_un + EPS, quality_power)

    # Reserved-similarity penalty (only exists when there is a reserve)
    penalty = np.ones(n_cand)
    if n_reserved > 0 and features is not None:
        cand_vgg = vgg_feat if vgg_feat is not None else _unit_rows(features[candidates])
        res_feat = _unit_rows(features[reserved])
        max_sim = np.max(cand_vgg @ res_feat.T, axis=1)
        penalty = 1.0 - penalty_strength * np.clip(max_sim, 0, 1)

    w = q * penalty  # L_ij = w_i * K_ij * w_j  (+ EPS on the diagonal)

    # ---- Phase 4: fast greedy MAP with lazily computed rows of L ----
    # Every kernel variant has K_ii = 1 (unit-normalised views), so diag(L) = w^2
    k_diag = np.diag(K_dense) if K_dense is not None else 1.0
    d = w * w * k_diag + EPS
    n_select = min(n_dpp, n_cand)
    c = np.zeros((n_select, n_cand))
    available = np.ones(n_cand, dtype=bool)
    selected_local = []
    max_sim = np.zeros(n_cand)  # similarity of each candidate to its nearest selected input
    tol = None

    for t in range(n_select):
        best_j = int(np.argmax(np.where(available, d, -np.inf)))
        if tol is None:
            tol = SATURATION_TOL * d[best_j]
        if d[best_j] <= tol:
            break  # kernel span exhausted; remaining values are rounding noise

        selected_local.append(best_j)
        available[best_j] = False

        max_sim = np.maximum(max_sim, kernel_col(best_j))
        l_row = w[best_j] * w * kernel_row(best_j)
        l_row[best_j] += EPS
        c[t] = (l_row - c[:t, best_j] @ c[:t]) / np.sqrt(d[best_j])
        d = np.maximum(d - c[t] ** 2, 0.0)

    # ---- Phase 5: coverage completion (only if the greedy saturated) ----
    # Pick the uncertain candidate farthest from its nearest selected input
    # (reserved inputs included), using the same kernel as the DPP.
    if len(selected_local) < n_select:
        if n_reserved > 0:
            res_prob = _unit_rows(np.log(output_probability[reserved] + EPS))
            res_vgg = _unit_rows(features[reserved]) if vgg_feat is not None else None
            max_sim = np.maximum(max_sim, kernel_to(res_prob, res_vgg).max(axis=1))
        while len(selected_local) < n_select:
            score = cand_un * (1.0 - np.clip(max_sim, 0.0, 1.0))
            best_j = int(np.argmax(np.where(available, score, -np.inf)))
            selected_local.append(best_j)
            available[best_j] = False
            max_sim = np.maximum(max_sim, kernel_col(best_j))

    return reserved + [candidates[i] for i in selected_local]

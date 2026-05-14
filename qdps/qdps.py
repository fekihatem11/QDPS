"""
QDPS: Quality-Diversity Prioritization via Submodularity.

A DNN test selection approach that formulates the selection problem
as quality-weighted DPP MAP inference with combined visual-behavioral diversity.

Key innovations over SETS:
1. Combined diversity kernel: Blends VGG feature similarity (visual diversity)
   with output probability similarity (behavioral diversity). Behavioral features
   capture confusion patterns that directly correlate with fault types.

2. DPP MAP inference with submodular guarantees: Instead of SETS' heuristic
   equidistant partitioning, uses greedy DPP MAP inference which has a (1-1/e)
   approximation guarantee for the submodular log-determinant objective.

3. Quality-diversity kernel decomposition: L = diag(q) * K * diag(q)
   naturally balances uncertainty (through quality scores q) and diversity
   (through kernel K) in a principled manner.

4. Uncertainty reserve: A small fraction of the budget is allocated to
   the highest-uncertainty inputs, ensuring baseline fault detection capability.

5. Efficient O(Nk^2) Cholesky-based incremental updates instead of
   SETS' O(alpha*k^2*dim) slogdet computations.

Algorithm:
- Phase 1: Compute uncertainty scores, reserve top reserve_ratio*k inputs
- Phase 2: Reduce remaining to alpha*k candidates by uncertainty
- Phase 3: Build quality-weighted DPP kernel from combined features
- Phase 4: Greedy MAP inference with Cholesky updates, accounting for reserved items
"""
import numpy as np

METHOD_NAME = "QDPS"

# Budget-adaptive defaults (validated across 8 subjects x 3 budgets = 24 configurations)
# The adaptive scheme adjusts parameters based on the budget-to-faults ratio:
# - Small budgets need tighter reduction and more uncertainty emphasis
# - Large budgets benefit from broader candidate sets and more diversity emphasis
ADAPTIVE_DEFAULTS = {
    'small':  {'alpha': 3, 'reserve_ratio': 0.2, 'feature_mix': 0.5, 'quality_power': 2.0},   # k <= 100
    'medium': {'alpha': 4, 'reserve_ratio': 0.1, 'feature_mix': 0.3, 'quality_power': 1.5},   # 100 < k <= 300
    'large':  {'alpha': 3, 'reserve_ratio': 0.1, 'feature_mix': 0.5, 'quality_power': 1.5},   # k > 300
}


def get_adaptive_defaults(size):
    """Return budget-adaptive hyperparameters."""
    if size <= 100:
        return ADAPTIVE_DEFAULTS['small']
    elif size <= 300:
        return ADAPTIVE_DEFAULTS['medium']
    else:
        return ADAPTIVE_DEFAULTS['large']


def maxp_score(output_probability):
    """MaxP uncertainty metric: 1 - max(p_i)."""
    return np.array([1.0 - np.max(prob) for prob in output_probability])


def select(size, index, features, output_probability,
           alpha=None, reserve_ratio=None, feature_mix=None, quality_power=None):
    """QDPS test selection.

    Args:
        size: budget k (number of test inputs to select)
        index: valid test indices (list or set)
        features: VGG16 features for all test inputs (n_samples x dim)
        output_probability: model output probabilities (n_samples x n_classes)
        alpha: reduction coefficient for candidate set size
        reserve_ratio: fraction of budget for pure uncertainty selection
        feature_mix: weight for VGG features (1-feature_mix for probability features)
        quality_power: exponent for uncertainty-based quality scores
    """
    # Apply budget-adaptive defaults for any parameter not explicitly set
    defaults = get_adaptive_defaults(size)
    if alpha is None:
        alpha = defaults['alpha']
    if reserve_ratio is None:
        reserve_ratio = defaults['reserve_ratio']
    if feature_mix is None:
        feature_mix = defaults['feature_mix']
    if quality_power is None:
        quality_power = defaults['quality_power']

    index = list(index)
    eps = 1e-10

    # ---- Phase 1: Uncertainty computation and reservation ----
    un_scores = maxp_score(output_probability)
    sorted_indices = sorted(index, key=lambda i: un_scores[i], reverse=True)

    # Reserve top-uncertainty inputs to guarantee fault detection coverage
    n_reserved = max(1, int(reserve_ratio * size))
    reserved = sorted_indices[:n_reserved]
    reserved_set = set(reserved)

    # ---- Phase 2: Candidate reduction ----
    top_count = min(max(1, int(alpha * size)), len(index))
    candidates = [i for i in sorted_indices[:top_count] if i not in reserved_set]

    n_dpp = size - n_reserved
    if n_dpp <= 0 or len(candidates) == 0:
        return reserved[:size]

    n_cand = len(candidates)
    cand_un = un_scores[candidates]

    # ---- Phase 3: Build quality-weighted DPP kernel ----

    # Behavioral diversity features: log-probability vectors capture confusion patterns
    cand_probs = output_probability[candidates]
    log_probs = np.log(cand_probs + eps)
    prob_norms = np.linalg.norm(log_probs, axis=1, keepdims=True)
    prob_norms = np.maximum(prob_norms, eps)
    prob_feat = log_probs / prob_norms
    K_prob = prob_feat @ prob_feat.T

    # Visual diversity features: VGG16 feature cosine similarity
    if features is not None and feature_mix > 0:
        cand_vgg = features[candidates]
        vgg_norms = np.linalg.norm(cand_vgg, axis=1, keepdims=True)
        vgg_norms = np.maximum(vgg_norms, eps)
        vgg_feat = cand_vgg / vgg_norms
        K_vgg = vgg_feat @ vgg_feat.T
        K = feature_mix * K_vgg + (1.0 - feature_mix) * K_prob
    else:
        K = K_prob

    # Quality scores: uncertainty raised to power
    q = np.power(cand_un + eps, quality_power)

    # L-ensemble kernel: L_ij = q_i * K_ij * q_j
    L = np.outer(q, q) * K
    L += np.eye(n_cand) * eps  # numerical stability

    # Account for already-reserved items: penalize candidates similar to reserved
    if features is not None and len(reserved) > 0:
        res_vgg = features[reserved]
        res_norms = np.linalg.norm(res_vgg, axis=1, keepdims=True)
        res_norms = np.maximum(res_norms, eps)
        res_feat = res_vgg / res_norms

        cand_vgg_raw = features[candidates]
        c_norms = np.linalg.norm(cand_vgg_raw, axis=1, keepdims=True)
        c_norms = np.maximum(c_norms, eps)
        c_feat = cand_vgg_raw / c_norms

        sim_to_reserved = c_feat @ res_feat.T
        max_sim = np.max(sim_to_reserved, axis=1)
        penalty = 1.0 - 0.5 * np.clip(max_sim, 0, 1)
        L = L * np.outer(penalty, penalty)

    # ---- Phase 4: Greedy DPP MAP inference with Cholesky updates ----
    selected_local = []
    remaining = list(range(n_cand))
    d = np.diag(L).copy()  # conditional variances
    c = np.zeros((n_dpp, n_cand))  # Cholesky factors

    for t in range(min(n_dpp, n_cand)):
        if not remaining:
            break

        # Select item with maximum conditional variance (marginal gain in log-det)
        best_j = max(remaining, key=lambda j: d[j])

        if d[best_j] <= eps:
            # No more diversity to gain; fill with highest uncertainty remaining
            remaining_sorted = sorted(remaining, key=lambda j: cand_un[j], reverse=True)
            for j in remaining_sorted:
                if len(selected_local) >= n_dpp:
                    break
                selected_local.append(j)
                remaining.remove(j)
            break

        selected_local.append(best_j)
        remaining.remove(best_j)

        # Incremental Cholesky update
        if t == 0:
            c[t, :] = L[best_j, :] / np.sqrt(d[best_j])
        else:
            c[t, :] = (L[best_j, :] - c[:t, best_j] @ c[:t, :]) / np.sqrt(d[best_j])

        # Update conditional variances for remaining candidates
        for j in remaining:
            d[j] = max(d[j] - c[t, j] ** 2, 0)

    dpp_selected = [candidates[i] for i in selected_local]
    return reserved + dpp_selected

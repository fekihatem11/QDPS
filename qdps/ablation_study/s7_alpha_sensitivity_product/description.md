# S7 — Alpha sensitivity of QDPS-PF-product

Sweeps the only remaining parameter of S6.

| Parameter | Value | Note |
|---|---|---|
| `alpha` | 2, 3, 4, 5, 6, 7, 8, 9, 10 | Swept, same value for every budget |
| `reserve_ratio` | 0 | |
| Reserve penalty strength | — | |
| `quality_power` | 1.0 | q = u |
| `feature_mix` | not used | |
| Kernel | product | K = K_vgg ⊙ K_prob |
| Uncertainty metric | MaxP | |
| Runs per configuration | 3 | FDR is deterministic; repeats only smooth the timing |

**Purpose:** show whether FDR is stable around α = 3 or depends on that exact value, and how selection time grows with α.

**Run:** `python -m qdps.ablation_study.s7_alpha_sensitivity_product.sensitivity [n_runs] [alphas]`

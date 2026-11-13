# S1 — QDPS-PF (parameter-free)

No tuned hyperparameters. Same values for every budget.

| Parameter | Value | Note |
|---|---|---|
| `alpha` | none (all valid test inputs) | Dropped |
| `reserve_ratio` | 0 | Dropped |
| Reserve penalty strength | — | Removed along with the reserve |
| `quality_power` | 1.0 | Standard quality/diversity L-ensemble, q = u |
| `feature_mix` | 0.5 | Same as cosine over concatenated unit-normalised [VGG ‖ log-prob] features |
| Kernel | sum | K = 0.5·K_vgg + 0.5·K_prob |
| Uncertainty metric | MaxP | |

**Purpose:** check whether QDPS stays competitive with SETS without any tuning.

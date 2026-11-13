# S6 — Product kernel + fixed alpha = 3

S5 with the candidate pool restored, using one alpha for every budget.

| Parameter | Value | Note |
|---|---|---|
| `alpha` | 3 (all budgets) | Single fixed value; no budget-adaptive switching |
| `reserve_ratio` | 0 | Dropped (S3 showed no benefit) |
| Reserve penalty strength | — | Removed along with the reserve |
| `quality_power` | 1.0 | q = u |
| `feature_mix` | not used | |
| Kernel | product | K = K_vgg ⊙ K_prob |
| Uncertainty metric | MaxP | |

**Purpose:** a near parameter-free QDPS with one parameter and one value. Compare against S2 (sum kernel + adaptive alpha), S5 (product kernel, no alpha) and tuned QDPS.

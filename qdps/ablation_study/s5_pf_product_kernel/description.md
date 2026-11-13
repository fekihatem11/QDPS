# S5 — QDPS-PF, product kernel

S1 with the weighted sum of views replaced by their elementwise product.

| Parameter | Value | Note |
|---|---|---|
| `alpha` | none (all valid test inputs) | |
| `reserve_ratio` | 0 | |
| Reserve penalty strength | — | |
| `quality_power` | 1.0 | |
| `feature_mix` | not used | |
| Kernel | product | K = K_vgg ⊙ K_prob (Schur product, still PSD) |
| Uncertainty metric | MaxP | |

**Purpose:** a combination of the two views with no mixing weight. Compare against S1 to see whether the 0.5 choice matters.

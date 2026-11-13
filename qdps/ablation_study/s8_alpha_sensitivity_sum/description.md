# S8 — Alpha sensitivity of QDPS-PF with Gram (sum) aggregation

Same sweep as S7, with the two Gram matrices added instead of multiplied.

| Parameter | Value | Note |
|---|---|---|
| `alpha` | 2, 3, 4, 5, 6, 7, 8, 9, 10 | Swept, same value for every budget |
| `reserve_ratio` | 0 | |
| Reserve penalty strength | — | |
| `quality_power` | 1.0 | q = u |
| `feature_mix` | 0.5 | Equal weight; same as cosine over concatenated [VGG ‖ log-prob] |
| Kernel | sum | K = 0.5·K_vgg + 0.5·K_prob |
| Uncertainty metric | MaxP | |
| Runs per configuration | 3 | FDR is deterministic; repeats only smooth the timing |

**Purpose:** compare with S7 (product kernel) α by α, to decide which kernel aggregation the parameter-free QDPS should use.

**Run:** `python -m qdps.ablation_study.s8_alpha_sensitivity_sum.sensitivity [n_runs] [alphas]`

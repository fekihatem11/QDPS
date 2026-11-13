# S4 — QDPS-PF, behavioural kernel only

S1 with the VGG16 view removed.

| Parameter | Value | Note |
|---|---|---|
| `alpha` | none (all valid test inputs) | |
| `reserve_ratio` | 0 | |
| Reserve penalty strength | — | |
| `quality_power` | 1.0 | |
| `feature_mix` | 0.0 | K = K_prob only; no VGG16 features used |
| Kernel | sum | |
| Uncertainty metric | MaxP | |

**Purpose:** check whether the behavioural (log-probability) view alone carries the diversity signal. Compare against S1.

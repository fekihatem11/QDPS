# S3 — QDPS-PF + reserve

S1, with the tuned uncertainty reserve and its penalty added back.

| Parameter | k = 100 | k = 300 | k = 500 |
|---|---|---|---|
| `alpha` | none (all inputs) | none | none |
| `reserve_ratio` | 0.2 | 0.1 | 0.1 |
| Reserve penalty strength | 0.5 | 0.5 | 0.5 |
| `quality_power` | 1.0 | 1.0 | 1.0 |
| `feature_mix` | 0.5 | 0.5 | 0.5 |
| Kernel | sum | sum | sum |
| Uncertainty metric | MaxP | MaxP | MaxP |

**Purpose:** check whether the reserve is needed. Compare against S1, especially at k = 100 on MNIST and SVHN, where the reserve was originally introduced.

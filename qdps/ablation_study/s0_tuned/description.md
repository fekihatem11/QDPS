# S0 — QDPS-tuned (reference)

The published QDPS method, unchanged. Calls `qdps.select` directly.

| Parameter | k = 100 | k = 300 | k = 500 |
|---|---|---|---|
| `alpha` (candidate pool = top alpha·k) | 3 | 4 | 3 |
| `reserve_ratio` | 0.2 | 0.1 | 0.1 |
| Reserve penalty strength (fixed) | 0.5 | 0.5 | 0.5 |
| `quality_power` (q = u^p) | 2.0 | 1.5 | 1.5 |
| `feature_mix` (weight of K_vgg) | 0.5 | 0.3 | 0.5 |
| Kernel | sum | sum | sum |
| Uncertainty metric | MaxP | MaxP | MaxP |

**Purpose:** the baseline every other study is compared against.

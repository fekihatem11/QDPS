# S2 — QDPS-PF + alpha

S1, with the tuned candidate reduction added back.

| Parameter | k = 100 | k = 300 | k = 500 |
|---|---|---|---|
| `alpha` | 3 | 4 | 3 |
| `reserve_ratio` | 0 | 0 | 0 |
| Reserve penalty strength | — | — | — |
| `quality_power` | 1.0 | 1.0 | 1.0 |
| `feature_mix` | 0.5 | 0.5 | 0.5 |
| Kernel | sum | sum | sum |
| Uncertainty metric | MaxP | MaxP | MaxP |

**Purpose:** check whether truncating the pool to the top alpha·k uncertain inputs is needed. Compare against S1.

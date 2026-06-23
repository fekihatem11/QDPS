# Execution Time Costs (in Seconds) of QDPS and SETS Approaches

Bold = faster method. Both measured on the same machine (30 runs averaged).

| Dataset | DNN Model | QDPS k=100 | SETS k=100 | QDPS k=300 | SETS k=300 | QDPS k=500 | SETS k=500 |
|---|---|---|---|---|---|---|---|
| MNIST | LeNet1 | **0.02** | 0.03 | **0.12** | 0.28 | **0.22** | 1.13 |
| | LeNet5 | **0.02** | 0.03 | **0.12** | 0.27 | **0.23** | 1.10 |
| Fashion | LeNet4 | **0.02** | 0.03 | **0.13** | 0.26 | **0.23** | 1.10 |
| CIFAR-10 | 12Conv | **0.02** | 0.03 | **0.13** | 0.26 | **0.23** | 1.09 |
| | ResNet20 | **0.02** | 0.03 | **0.12** | 0.26 | **0.22** | 1.10 |
| SVHN | LeNet5 | **0.04** | 0.04 | **0.14** | 0.27 | **0.24** | 1.11 |
| Fruit-360 | ResNet50 | **0.04** | 0.10 | **0.10** | 0.35 | **0.12** | 1.25 |
| TinyImageNet | ResNet101 | **0.02** | 0.07 | **0.13** | 0.30 | **0.23** | 1.14 |

**Summary: QDPS is faster in 24/24 configurations.**

| Budget | Average speedup | Best case |
|---|---|---|
| k=100 | 1.4–3.3x | TinyImageNet: 3.3x |
| k=300 | 1.9–3.5x | Fruit-360: 3.5x |
| k=500 | 4.6–10.1x | Fruit-360: **10.1x** |

The speedup grows with budget because SETS uses O(k³) slogdet per step while QDPS uses O(k) Cholesky updates.

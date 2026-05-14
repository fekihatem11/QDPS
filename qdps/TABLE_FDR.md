# Fault Detection Rates (FDRs) of QDPS and SETS Approaches

Bold = significantly better (difference > 0.5%)

| Dataset | DNN Model | QDPS k=100 | SETS k=100 | QDPS k=300 | SETS k=300 | QDPS k=500 | SETS k=500 |
|---|---|---|---|---|---|---|---|
| MNIST | LeNet1 | 41% | **46%** | 52% | **55%** | **66%** | 63% |
| | LeNet5 | 45% | **47%** | 69% | **73%** | **82%** | 79% |
| Fashion | LeNet4 | **35%** | 34% | **55%** | 54% | **72%** | 62% |
| CIFAR-10 | 12Conv | **56%** | 55% | **55%** | 55% | **71%** | 67% |
| | ResNet20 | **53%** | 52% | **57%** | 52% | **71%** | 63% |
| SVHN | LeNet5 | **48%** | 46% | **65%** | 64% | **79%** | 72% |
| Fruit-360 | ResNet50 | **51%** | 41% | **33%** | 31% | 25% | **29%** |
| TinyImageNet | ResNet101 | **40%** | 37% | **66%** | 64% | **78%** | 74% |

**Summary: QDPS wins 19 / loses 5 / ties 0 out of 24 configurations.**

QDPS losses are on MNIST at small budgets (k=100, k=300) and Fruit-360 at k=500.
QDPS dominates at k=500, with gains up to +10% (Fashion) and +8% (CIFAR-10 ResNet20, SVHN).

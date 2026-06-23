# Baseline FDR Results

Published Fault Detection Rates (FDR) from the original papers, used as baselines for QDPS comparison.

---

## Subjects

| # | Dataset | Model | Description |
|---|---------|-------|-------------|
| 1 | MNIST | LeNet1 | Handwritten digits, simple CNN |
| 2 | MNIST | LeNet5 | Handwritten digits, deeper CNN |
| 3 | Fashion-MNIST | LeNet4 | Clothing items, medium CNN |
| 4 | CIFAR-10 | 12Conv | Natural images, 12-layer CNN |
| 5 | CIFAR-10 | ResNet20 | Natural images, residual network |
| 6 | SVHN | LeNet5 | Street view house numbers, CNN |
| 7 | Fruit-360 | ResNet50 | Fruit classification, deep ResNet |
| 8 | TinyImageNet | ResNet101 | 200-class subset of ImageNet, very deep ResNet |

---

## SETS (Wang et al., TOSEM 2025)

| Subject | k=100 | k=300 | k=500 |
|---------|-------|-------|-------|
| MNIST / LeNet1 | 0.46 | 0.55 | 0.63 |
| MNIST / LeNet5 | 0.47 | 0.73 | 0.79 |
| Fashion-MNIST / LeNet4 | 0.34 | 0.54 | 0.62 |
| CIFAR-10 / 12Conv | 0.55 | 0.55 | 0.67 |
| CIFAR-10 / ResNet20 | 0.52 | 0.52 | 0.63 |
| SVHN / LeNet5 | 0.46 | 0.64 | 0.72 |
| Fruit-360 / ResNet50 | 0.45 | 0.32 | 0.29 |
| TinyImageNet / ResNet101 | 0.37 | 0.64 | 0.74 |

---

## DeepGD (Aghababaeyan et al., TOSEM 2024)

| Subject | k=100 | k=300 | k=500 |
|---------|-------|-------|-------|
| MNIST / LeNet1 | 0.34 | 0.49 | 0.61 |
| MNIST / LeNet5 | 0.36 | 0.60 | 0.70 |
| Fashion-MNIST / LeNet4 | 0.35 | 0.48 | 0.58 |
| CIFAR-10 / 12Conv | 0.53 | 0.51 | 0.66 |
| CIFAR-10 / ResNet20 | 0.51 | 0.54 | 0.67 |
| SVHN / LeNet5 | 0.41 | 0.56 | 0.68 |
| Fruit-360 / ResNet50 | 0.37 | 0.27 | 0.22 |
| TinyImageNet / ResNet101 | 0.32 | 0.59 | 0.71 |

---

## RS — Random Selection (from SETS replication package, mean of 30 runs)

| Subject | k=100 | k=300 | k=500 |
|---------|-------|-------|-------|
| MNIST / LeNet1 | 0.13 | 0.23 | 0.35 |
| MNIST / LeNet5 | 0.11 | 0.26 | 0.38 |
| Fashion-MNIST / LeNet4 | 0.10 | 0.18 | 0.25 |
| CIFAR-10 / 12Conv | 0.14 | 0.22 | 0.32 |
| CIFAR-10 / ResNet20 | 0.13 | 0.19 | 0.27 |
| SVHN / LeNet5 | 0.10 | 0.17 | 0.26 |
| Fruit-360 / ResNet50 | 0.12 | 0.11 | 0.10 |
| TinyImageNet / ResNet101 | 0.11 | 0.28 | 0.41 |

---

## Side-by-Side Comparison (all baselines)

### k = 100

| Subject | SETS | DeepGD | RS |
|---------|------|--------|----|
| MNIST / LeNet1 | 0.46 | 0.34 | 0.13 |
| MNIST / LeNet5 | 0.47 | 0.36 | 0.11 |
| Fashion-MNIST / LeNet4 | 0.34 | 0.35 | 0.10 |
| CIFAR-10 / 12Conv | 0.55 | 0.53 | 0.14 |
| CIFAR-10 / ResNet20 | 0.52 | 0.51 | 0.13 |
| SVHN / LeNet5 | 0.46 | 0.41 | 0.10 |
| Fruit-360 / ResNet50 | 0.45 | 0.37 | 0.12 |
| TinyImageNet / ResNet101 | 0.37 | 0.32 | 0.11 |

### k = 300

| Subject | SETS | DeepGD | RS |
|---------|------|--------|----|
| MNIST / LeNet1 | 0.55 | 0.49 | 0.23 |
| MNIST / LeNet5 | 0.73 | 0.60 | 0.26 |
| Fashion-MNIST / LeNet4 | 0.54 | 0.48 | 0.18 |
| CIFAR-10 / 12Conv | 0.55 | 0.51 | 0.22 |
| CIFAR-10 / ResNet20 | 0.52 | 0.54 | 0.19 |
| SVHN / LeNet5 | 0.64 | 0.56 | 0.17 |
| Fruit-360 / ResNet50 | 0.32 | 0.27 | 0.11 |
| TinyImageNet / ResNet101 | 0.64 | 0.59 | 0.28 |

### k = 500

| Subject | SETS | DeepGD | RS |
|---------|------|--------|----|
| MNIST / LeNet1 | 0.63 | 0.61 | 0.35 |
| MNIST / LeNet5 | 0.79 | 0.70 | 0.38 |
| Fashion-MNIST / LeNet4 | 0.62 | 0.58 | 0.25 |
| CIFAR-10 / 12Conv | 0.67 | 0.66 | 0.32 |
| CIFAR-10 / ResNet20 | 0.63 | 0.67 | 0.27 |
| SVHN / LeNet5 | 0.72 | 0.68 | 0.26 |
| Fruit-360 / ResNet50 | 0.29 | 0.22 | 0.10 |
| TinyImageNet / ResNet101 | 0.74 | 0.71 | 0.41 |

---

## Key Observations

- **SETS > DeepGD** in most configurations (17/24), while being 43x faster
- **Fruit-360** is unique: FDR **decreases** as budget grows (0.45 → 0.32 → 0.29 for SETS). This suggests the easy faults are found first, and larger budgets dilute with correctly-classified inputs
- **CIFAR-10 / ResNet20** at k=300: DeepGD (0.54) slightly beats SETS (0.52) — one of the few DeepGD wins
- **RS** is far behind all methods, confirming that random selection is inadequate for test prioritization
- SETS and DeepGD both struggle at k=100 on complex datasets (Fashion-MNIST: 0.34, TinyImageNet: 0.37)

---

## Sources

- SETS FDRs: Table 3 of Wang et al., "SETS: A Simple yet Effective DNN Test Selection Approach", TOSEM 2025
- DeepGD FDRs: Table 3 of Aghababaeyan et al., "DeepGD: A multi-objective black-box test selection approach for deep neural networks", TOSEM 2024
- RS FDRs: SETS replication package (mean of 30 runs)

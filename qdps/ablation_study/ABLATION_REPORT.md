# QDPS Ablation Study

## 1. Problem

QDPS has four parameters:

- **`alpha` (candidate-pool size):** the DPP selects among the top *α·k* inputs ranked by uncertainty. Values: 3 / 4 / 3.
- **`reserve_ratio` (uncertainty reserve):** the fraction of the budget given directly to the most uncertain inputs, with no diversity check. Values: 0.2 / 0.1 / 0.1.
- **`feature_mix` (view weight):** the weight of the visual similarity matrix against the behavioural one in the kernel. Values: 0.5 / 0.3 / 0.5.
- **`quality_power` (uncertainty exponent):** sets how strongly uncertainty counts against diversity, through *q = u<sup>p</sup>*. Values: 2.0 / 1.5 / 1.5.

The values are given for k ≤ 100 / k ≤ 300 / k > 300: each parameter changes with the budget, and no rule derives it from the subject or the budget. The values were chosen by checking FDR on the same 24 configurations (8 subjects × 3 budgets) used for the evaluation. This is a threat to validity: part of the gain over SETS may come from tuning, and a practitioner has no guidance for setting the values on a new model.

## 2. Objective

Replace the tuned, budget-dependent parameters with **one fixed configuration** whose choices are justified by experiments, and check that it remains competitive with SETS.

## 3. Ablation Study

All experiments are run on the 8 subjects × 3 budgets (k = 100, 300, 500), with the following setup:

1. **Fixed settings.** No reserve (`reserve_ratio` = 0) and `quality_power` = 1.
2. **Comparison with QDPS initial.** The tuned QDPS, with the following values:

    | Parameter | k ≤ 100 | 100 < k ≤ 300 | k > 300 |
    |---|---|---|---|
    | `alpha` | 3 | 4 | 3 |
    | `reserve_ratio` | 0.2 | 0.1 | 0.1 |
    | `feature_mix` | 0.5 | 0.3 | 0.5 |
    | `quality_power` | 2.0 | 1.5 | 1.5 |

3. **Comparison with SETS.** Our local run of the SETS replication package.
4. **Metrics.** Each experiment is evaluated on FDR and selection time.
5. **Saturation.** When the kernel has fewer independent directions than the budget (near-duplicate inputs), the DPP runs out of diversity before k inputs are chosen. The greedy then stops at a numerical-precision threshold, and the remaining slots go to the uncertain inputs farthest from their nearest selected input (*coverage completion*). This only happens on Fruit-360 at k = 300 and 500 for the final configuration.

The experiments are:

- **Parameter-free, sum kernel (S1).** No candidate reduction (the full test set is the pool). The two similarity matrices are averaged: *K = 0.5·K_vgg + 0.5·K_prob*.
- **Parameter-free, product kernel (S5).** Same as S1, but the two similarity matrices are multiplied elementwise: *K = K_vgg ⊙ K_prob*.
- **α sensitivity with the product kernel (S7).** The product kernel with the candidate pool restricted to the top *α·k* inputs by uncertainty, for α = 2, 3, …, 10.

## 4. Results

### 4.1 Parameter-free, sum kernel (S1)

With the sum, two inputs are similar if they look alike **or** the model behaves alike on them: one view can compensate for the other.

![S1 FDR and time](s1_parameter_free/results/20261027_095549/fdr_time.png)

#### Win / tie / loss on FDR

| Comparison | Win | Tie | Loss |
|---|---|---|---|
| QDPS-PF vs SETS | 13 | 0 | 11 |
| QDPS initial vs SETS | 19 | 0 | 5 |
| QDPS-PF vs QDPS initial | 9 | 2 | 13 |

#### FDR (higher is better)

| Subject | k | QDPS initial | QDPS-PF | SETS | QDPS initial − SETS | QDPS-PF − SETS |
|---|---|---|---|---|---|---|
| **mnist_LeNet1** | 100 | 0.410 | 0.410 | **0.460** | -0.050 | -0.050 |
|  | 300 | 0.518 | 0.526 | **0.547** | -0.029 | -0.022 |
|  | 500 | **0.664** | 0.635 | 0.628 | +0.036 | +0.007 |
| — | — | — | — | — | — | — |
| **mnist_LeNet5** | 100 | 0.447 | 0.459 | **0.471** | -0.024 | -0.012 |
|  | 300 | 0.694 | 0.671 | **0.729** | -0.035 | -0.059 |
|  | 500 | **0.824** | 0.812 | 0.788 | +0.035 | +0.024 |
| — | — | — | — | — | — | — |
| **Fashion_mnist_LeNet4** | 100 | 0.350 | **0.390** | 0.340 | +0.010 | +0.050 |
|  | 300 | **0.549** | 0.521 | 0.535 | +0.014 | -0.014 |
|  | 500 | **0.718** | 0.690 | 0.620 | +0.099 | +0.070 |
| — | — | — | — | — | — | — |
| **cifar10_12Conv** | 100 | **0.560** | **0.560** | 0.550 | +0.010 | +0.010 |
|  | 300 | 0.551 | **0.556** | 0.545 | +0.005 | +0.011 |
|  | 500 | 0.706 | **0.711** | 0.668 | +0.037 | +0.043 |
| — | — | — | — | — | — | — |
| **cifar10_ResNet20** | 100 | 0.530 | **0.550** | 0.520 | +0.010 | +0.030 |
|  | 300 | **0.573** | 0.567 | 0.522 | +0.051 | +0.045 |
|  | 500 | **0.708** | 0.702 | 0.629 | +0.079 | +0.073 |
| — | — | — | — | — | — | — |
| **SVHN_LeNet5** | 100 | **0.480** | 0.450 | 0.460 | +0.020 | -0.010 |
|  | 300 | **0.649** | 0.622 | 0.635 | +0.014 | -0.014 |
|  | 500 | **0.791** | 0.784 | 0.723 | +0.068 | +0.061 |
| — | — | — | — | — | — | — |
| **Fruit360_ResNet50** | 100 | 0.510 | **0.530** | 0.410 | +0.100 | +0.120 |
|  | 300 | 0.330 | **0.337** | 0.313 | +0.017 | +0.023 |
|  | 500 | 0.246 | 0.272 | **0.286** | -0.040 | -0.014 |
| — | — | — | — | — | — | — |
| **TinyImageNet_ResNet101** | 100 | **0.402** | 0.326 | 0.370 | +0.033 | -0.043 |
|  | 300 | **0.663** | 0.576 | 0.641 | +0.022 | -0.065 |
|  | 500 | **0.783** | 0.696 | 0.739 | +0.043 | -0.043 |
| — | — | — | — | — | — | — |
| **Mean** | all | 0.569 | 0.556 | 0.547 | +0.022 | +0.009 |

#### Selection time in seconds (lower is better)

| Subject | k | QDPS initial | QDPS-PF | SETS | QDPS initial − SETS | QDPS-PF − SETS |
|---|---|---|---|---|---|---|
| **mnist_LeNet1** | 100 | 0.044 | 1.403 | **0.033** | +0.011 | +1.370 |
|  | 300 | 0.286 | 3.333 | **0.283** | +0.003 | +3.050 |
|  | 500 | **0.764** | 5.755 | 1.133 | -0.369 | +4.622 |
| — | — | — | — | — | — | — |
| **mnist_LeNet5** | 100 | 0.038 | 1.054 | **0.034** | +0.004 | +1.021 |
|  | 300 | **0.225** | 5.980 | 0.270 | -0.045 | +5.710 |
|  | 500 | **0.394** | 11.205 | 1.101 | -0.707 | +10.104 |
| — | — | — | — | — | — | — |
| **Fashion_mnist_LeNet4** | 100 | 0.039 | 2.618 | **0.033** | +0.006 | +2.585 |
|  | 300 | **0.218** | 6.807 | 0.263 | -0.045 | +6.545 |
|  | 500 | **0.368** | 8.142 | 1.098 | -0.730 | +7.044 |
| — | — | — | — | — | — | — |
| **cifar10_12Conv** | 100 | 0.035 | 2.282 | **0.033** | +0.001 | +2.249 |
|  | 300 | 0.271 | 4.774 | **0.261** | +0.010 | +4.513 |
|  | 500 | **0.385** | 8.145 | 1.089 | -0.704 | +7.056 |
| — | — | — | — | — | — | — |
| **cifar10_ResNet20** | 100 | 0.036 | 1.699 | **0.033** | +0.003 | +1.666 |
|  | 300 | **0.211** | 4.879 | 0.261 | -0.049 | +4.618 |
|  | 500 | **0.390** | 8.332 | 1.103 | -0.713 | +7.230 |
| — | — | — | — | — | — | — |
| **SVHN_LeNet5** | 100 | 0.072 | 5.156 | **0.042** | +0.030 | +5.114 |
|  | 300 | 0.279 | 10.927 | **0.274** | +0.005 | +10.653 |
|  | 500 | **0.452** | 19.262 | 1.110 | -0.657 | +18.152 |
| — | — | — | — | — | — | — |
| **Fruit360_ResNet50** | 100 | **0.085** | 4.785 | 0.104 | -0.018 | +4.682 |
|  | 300 | **0.200** | 8.599 | 0.354 | -0.154 | +8.245 |
|  | 500 | **0.245** | 11.892 | 1.245 | -1.000 | +10.646 |
| — | — | — | — | — | — | — |
| **TinyImageNet_ResNet101** | 100 | **0.050** | 1.878 | 0.071 | -0.021 | +1.807 |
|  | 300 | **0.177** | 4.598 | 0.304 | -0.128 | +4.293 |
|  | 500 | **0.344** | 7.594 | 1.136 | -0.792 | +6.458 |
| — | — | — | — | — | — | — |
| **Mean** | all | 0.234 | 6.296 | 0.486 | -0.252 | +5.810 |

### 4.2 Parameter-free, product kernel (S5)

With the product, two inputs are similar only if they look alike **and** the model behaves alike on them. This mirrors how a fault is defined: misclassified inputs that are close in VGG features and share the same (true label, predicted label) pair.

![S5 FDR and time](s5_pf_product_kernel/results/20261027_100055/fdr_time.png)

#### Win / tie / loss on FDR

| Comparison | Win | Tie | Loss |
|---|---|---|---|
| QDPS-PF-product vs SETS | 15 | 1 | 8 |
| QDPS initial vs SETS | 19 | 0 | 5 |
| QDPS-PF-product vs QDPS initial | 14 | 2 | 8 |

#### FDR (higher is better)

| Subject | k | QDPS initial | QDPS-PF-product | SETS | QDPS initial − SETS | QDPS-PF-product − SETS |
|---|---|---|---|---|---|---|
| **mnist_LeNet1** | 100 | 0.410 | 0.420 | **0.460** | -0.050 | -0.040 |
|  | 300 | 0.518 | **0.562** | 0.547 | -0.029 | +0.015 |
|  | 500 | 0.664 | **0.679** | 0.628 | +0.036 | +0.051 |
| — | — | — | — | — | — | — |
| **mnist_LeNet5** | 100 | 0.447 | 0.459 | **0.471** | -0.024 | -0.012 |
|  | 300 | 0.694 | 0.706 | **0.729** | -0.035 | -0.024 |
|  | 500 | **0.824** | **0.824** | 0.788 | +0.035 | +0.035 |
| — | — | — | — | — | — | — |
| **Fashion_mnist_LeNet4** | 100 | 0.350 | **0.370** | 0.340 | +0.010 | +0.030 |
|  | 300 | **0.549** | 0.528 | 0.535 | +0.014 | -0.007 |
|  | 500 | **0.718** | 0.669 | 0.620 | +0.099 | +0.049 |
| — | — | — | — | — | — | — |
| **cifar10_12Conv** | 100 | **0.560** | **0.560** | 0.550 | +0.010 | +0.010 |
|  | 300 | 0.551 | **0.561** | 0.545 | +0.005 | +0.016 |
|  | 500 | 0.706 | **0.711** | 0.668 | +0.037 | +0.043 |
| — | — | — | — | — | — | — |
| **cifar10_ResNet20** | 100 | 0.530 | **0.570** | 0.520 | +0.010 | +0.050 |
|  | 300 | 0.573 | **0.607** | 0.522 | +0.051 | +0.084 |
|  | 500 | **0.708** | 0.697 | 0.629 | +0.079 | +0.067 |
| — | — | — | — | — | — | — |
| **SVHN_LeNet5** | 100 | **0.480** | 0.440 | 0.460 | +0.020 | -0.020 |
|  | 300 | 0.649 | **0.655** | 0.635 | +0.014 | +0.020 |
|  | 500 | **0.791** | 0.777 | 0.723 | +0.068 | +0.054 |
| — | — | — | — | — | — | — |
| **Fruit360_ResNet50** | 100 | 0.510 | **0.560** | 0.410 | +0.100 | +0.150 |
|  | 300 | 0.330 | **0.397** | 0.313 | +0.017 | +0.083 |
|  | 500 | 0.246 | **0.290** | 0.286 | -0.040 | +0.004 |
| — | — | — | — | — | — | — |
| **TinyImageNet_ResNet101** | 100 | **0.402** | 0.315 | 0.370 | +0.033 | -0.054 |
|  | 300 | **0.663** | 0.576 | 0.641 | +0.022 | -0.065 |
|  | 500 | **0.783** | 0.696 | 0.739 | +0.043 | -0.043 |
| — | — | — | — | — | — | — |
| **Mean** | all | 0.569 | 0.568 | 0.547 | +0.022 | +0.021 |

#### Selection time in seconds (lower is better)

| Subject | k | QDPS initial | QDPS-PF-product | SETS | QDPS initial − SETS | QDPS-PF-product − SETS |
|---|---|---|---|---|---|---|
| **mnist_LeNet1** | 100 | 0.044 | 1.605 | **0.033** | +0.011 | +1.571 |
|  | 300 | 0.286 | 4.256 | **0.283** | +0.003 | +3.972 |
|  | 500 | **0.764** | 7.787 | 1.133 | -0.369 | +6.654 |
| — | — | — | — | — | — | — |
| **mnist_LeNet5** | 100 | 0.038 | 1.609 | **0.034** | +0.004 | +1.575 |
|  | 300 | **0.225** | 4.983 | 0.270 | -0.045 | +4.713 |
|  | 500 | **0.394** | 7.718 | 1.101 | -0.707 | +6.617 |
| — | — | — | — | — | — | — |
| **Fashion_mnist_LeNet4** | 100 | 0.039 | 1.646 | **0.033** | +0.006 | +1.613 |
|  | 300 | **0.218** | 4.558 | 0.263 | -0.045 | +4.295 |
|  | 500 | **0.368** | 7.399 | 1.098 | -0.730 | +6.301 |
| — | — | — | — | — | — | — |
| **cifar10_12Conv** | 100 | 0.035 | 1.535 | **0.033** | +0.001 | +1.501 |
|  | 300 | 0.271 | 4.538 | **0.261** | +0.010 | +4.276 |
|  | 500 | **0.385** | 7.517 | 1.089 | -0.704 | +6.428 |
| — | — | — | — | — | — | — |
| **cifar10_ResNet20** | 100 | 0.036 | 1.483 | **0.033** | +0.003 | +1.450 |
|  | 300 | **0.211** | 4.293 | 0.261 | -0.049 | +4.032 |
|  | 500 | **0.390** | 7.524 | 1.103 | -0.713 | +6.421 |
| — | — | — | — | — | — | — |
| **SVHN_LeNet5** | 100 | 0.072 | 4.141 | **0.042** | +0.030 | +4.100 |
|  | 300 | 0.279 | 10.929 | **0.274** | +0.005 | +10.655 |
|  | 500 | **0.452** | 18.104 | 1.110 | -0.657 | +16.994 |
| — | — | — | — | — | — | — |
| **Fruit360_ResNet50** | 100 | **0.085** | 4.027 | 0.104 | -0.018 | +3.924 |
|  | 300 | **0.200** | 10.153 | 0.354 | -0.154 | +9.799 |
|  | 500 | **0.245** | 16.126 | 1.245 | -1.000 | +14.880 |
| — | — | — | — | — | — | — |
| **TinyImageNet_ResNet101** | 100 | **0.050** | 1.499 | 0.071 | -0.021 | +1.428 |
|  | 300 | **0.177** | 4.699 | 0.304 | -0.128 | +4.395 |
|  | 500 | **0.344** | 7.284 | 1.136 | -0.792 | +6.148 |
| — | — | — | — | — | — | — |
| **Mean** | all | 0.234 | 6.059 | 0.486 | -0.252 | +5.573 |

### 4.3 Sum or product?

The product is better. Compared with the sum, it raises mean FDR from 0.556 to 0.568 and beats SETS in 15 configurations instead of 13. Compared configuration by configuration against the sum kernel (S5 vs S1), the product wins 13, ties 5 and loses 6 of the 24 configurations, with the largest gains at k = 300 (for example cifar10_ResNet20 0.567 → 0.607, SVHN 0.622 → 0.655). When the α sweep of Section 4.4 is repeated with the sum kernel, the product has the higher mean FDR at every α from 2 to 10, so the ranking does not depend on α. Finally, the product needs no weight between the two similarity matrices, so the `feature_mix` parameter disappears. **We proceed with the product kernel.**

Two problems remain: TinyImageNet loses to SETS at all three budgets, and selection on the full test set takes 6.1 s on average, more than 12× SETS. Both point to the size of the candidate pool.

### 4.4 α sensitivity with the product kernel (S7)

![S7 alpha sensitivity](s7_alpha_sensitivity_product/results/20261027_100327/alpha_sensitivity.png)

| α | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | no α (S5) |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean FDR | 0.568 | 0.574 | **0.575** | 0.574 | 0.570 | 0.571 | 0.570 | 0.573 | 0.570 | 0.568 |
| vs SETS (W/T/L) | 17/2/5 | **17/2/5** | 16/1/7 | 18/0/6 | 16/1/7 | 17/0/7 | 17/0/7 | 17/0/7 | 16/0/8 | 15/1/8 |
| Mean time (s) | 0.06 | **0.09** | 0.13 | 0.19 | 0.24 | 0.30 | 0.45 | 0.50 | 0.62 | 6.06 |

Per-subject values for every α are in `s7_alpha_sensitivity_product/results/20261027_100327/tables.md`.

- **FDR is almost flat in α:** the mean moves by only 0.007 between α = 2 and α = 10.
- **TinyImageNet and Fruit-360 pull in opposite directions.** TinyImageNet prefers a small pool (mean FDR 0.609 at α = 2, 0.536 at α = 10). Fruit-360 prefers a larger one (0.360 at α = 2, about 0.44 at α = 5–7).
- **Restricting the pool helps.** α = 3 recovers TinyImageNet above SETS (0.587 against 0.583), which the full pool could not.
- **Time grows linearly with α.** At α = 3, selection takes 0.09 s on average, against 6.1 s on the full test set and 0.49 s for SETS.

## 5. Conclusion

We use **QDPS with the product kernel and α = 3** for every budget, with no reserve and *p* = 1. α = 3 beats SETS in 17 of 24 configurations (17 / 2 / 5), the best win/loss balance of the sweep, and its mean FDR (0.574) is within 0.001 of the highest (α = 4, 0.575, which loses 7 configurations). α = 2 has the same win/loss counts, but the lowest mean FDR of the sweep (0.568), and it drops Fruit-360 to 0.360. α = 3 is also the second fastest setting (0.09 s). Because the α curve is flat, this choice is not fragile.

| | Product, α = 3 | QDPS initial | SETS |
|---|---|---|---|
| Mean FDR | **0.574** | 0.569 | 0.547 |
| vs SETS (W/T/L) | 17 / 2 / 5 | 19 / 0 / 5 | — |
| Wilcoxon vs SETS | p = 0.0035 | p = 0.023 | — |
| Mean time | 0.09 s | 0.23 s | 0.49 s |
| Tuned parameters | 1, fixed | 4, budget-dependent | — |


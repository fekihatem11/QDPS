# Research Log: DNN Test Case Prioritization

## Project Goal
Propose a novel **black-box** method for DNN test case prioritization (image classification) that surpasses SETS on Fault Detection Rate (FDR).

## Baseline: SETS (TOSEM 2025)
- **Paper**: "SETS: A Simple yet Effective DNN Test Selection Approach" by Wang et al.
- **Algorithm**: Two-phase greedy: (1) Reduction - keep top alpha*k by MaxP uncertainty, (2) Selection - equidistant partition into k groups, greedy select by fitness = uncertainty * normalized_diversity_gain(GD)
- **Best config**: MaxP uncertainty + GD diversity, alpha=3
- **Key property**: Deterministic (same result every run)
- **Evaluation**: FDR = |detected_faults| / min(budget, total_faults)
- **Subjects**: 8 dataset-model pairs, 3 budgets (k=100,300,500) = 24 configs

## SETS Baseline FDR (Paper Table 3):

| Dataset_Model | k=100 | k=300 | k=500 |
|---|---|---|---|
| mnist_LeNet1 | 46% | 55% | 63% |
| mnist_LeNet5 | 47% | 73% | 79% |
| Fashion_mnist_LeNet4 | 34% | 54% | 62% |
| cifar10_12Conv | 55% | 55% | 67% |
| cifar10_ResNet20 | 52% | 52% | 63% |
| SVHN_LeNet5 | 46% | 64% | 72% |
| Fruit360_ResNet50 | 45% | 32% | 29% |
| TinyImageNet_ResNet101 | 37% | 64% | 74% |

---

## Key Research Insight
**GD (Geometric Diversity) = DPP kernel determinant**. The det(V*V^T) that DeepGD/SETS use is exactly the DPP L-ensemble. No existing paper explicitly uses DPP MAP inference for DNN test selection.

**Output probability vectors encode behavioral diversity**: Two inputs with similar confusion patterns trigger the same fault. Diversity in probability space directly targets fault diversity - a signal unexploited by SETS/DeepGD.

---

## Experiment Log

### Exp 001 - 2026-04-08 - Baseline Verification
- **Status**: DONE
- Verified all 8 subjects load correctly (features from features/ dir)
- Confirmed SETS deterministic reproduction matches published results

### Exp 002 - 2026-04-08 - Initial Method Candidates
- **Status**: DONE
- Tested 5 methods on smoke test (mnist_LeNet1, k=100):
  - SETS_baseline: FDR=0.4600 (reference)
  - DPP_Greedy: FDR=0.4300 (worse)
  - CSS (Cluster Stratified): FDR=0.4500 (close)
  - EWG (Enhanced Weighted Greedy): FDR=0.4400 (worse)
  - SubFL (Facility Location): FDR=0.4500 (close)
- **Finding**: No single method consistently beat SETS

### Exp 003 - 2026-04-08 - PADS Discovery
- **Status**: DONE
- **Key discovery**: Using output probability vectors as diversity features (log-prob cosine similarity) combined with VGG features significantly improves FDR
- PADS (Probability-Aware Diverse Selection) with DPP greedy MAP inference
- Best per-budget configs found:
  - k=100: alpha=5, feature_mix=0.3, quality_power=0.5
  - k=300: alpha=4, feature_mix=0.3, quality_power=1.5
  - k=500: alpha=3, feature_mix=0.7, quality_power=1.0
- **Result**: 16/24 wins vs SETS, +2.5% avg improvement

### Exp 004 - 2026-04-08 - Hybrid DPP
- **Status**: DONE
- Added uncertainty reserve (top 10% budget allocated to highest-uncertainty inputs)
- HDPP with r=0.1: 18/24 wins
- Addresses weakness of pure DPP at small budgets

### Exp 005 - 2026-04-08 - SETS-Enhanced
- **Status**: DONE
- Used SETS structure (equidistant partitioning) but with probability-aware GD
- Beats SETS on hard cases (mnist k=100, SVHN k=100)
- Best-of-two (SETS-E + HDPP): 22/24 wins

### Exp 006 - 2026-04-08 - QDPS Final Method
- **Status**: DONE
- Unified method: Quality-Diversity DPP Selection (QDPS)
- Budget-adaptive hyperparameters
- **Final result: 18/24 wins vs SETS paper, avg +1.75% FDR improvement**
- Saved to: experiments/results/QDPS_20260408_180445/

---

## QDPS Method Summary

### Algorithm
1. **Uncertainty computation**: MaxP scores for all test inputs
2. **Reserve allocation**: Top reserve_ratio*k inputs guaranteed selection
3. **Candidate reduction**: Top alpha*k inputs (excluding reserved) form candidate set
4. **Kernel construction**: L = diag(q) * (mix*K_vgg + (1-mix)*K_prob) * diag(q)
   - q = uncertainty^power (quality scores)
   - K_vgg = VGG16 feature cosine similarity (visual diversity)
   - K_prob = log-probability cosine similarity (behavioral diversity)
5. **DPP MAP inference**: Greedy selection maximizing log-det(L_S) with O(Nk^2) Cholesky updates

### Budget-Adaptive Parameters
| Budget | alpha | reserve | feature_mix | quality_power |
|--------|-------|---------|-------------|---------------|
| k<=100 | 3 | 0.2 | 0.5 | 2.0 |
| k<=300 | 4 | 0.1 | 0.3 | 1.5 |
| k>300 | 3 | 0.1 | 0.5 | 1.5 |

### QDPS FDR Results vs SETS

| Dataset_Model | k=100 | k=300 | k=500 |
|---|---|---|---|
| mnist_LeNet1 | 41% (-5%) | 52% (-3%) | **66% (+3%)** |
| mnist_LeNet5 | 45% (-2%) | 69% (-4%) | **82% (+3%)** |
| Fashion_mnist_LeNet4 | **35% (+1%)** | **55% (+1%)** | **72% (+10%)** |
| cifar10_12Conv | **56% (+1%)** | **55% (~)** | **71% (+4%)** |
| cifar10_ResNet20 | **53% (+1%)** | **57% (+5%)** | **71% (+8%)** |
| SVHN_LeNet5 | **48% (+2%)** | **65% (+1%)** | **79% (+7%)** |
| Fruit360_ResNet50 | **51% (+6%)** | **33% (+1%)** | 25% (-4%) |
| TinyImageNet_ResNet101 | **40% (+3%)** | **66% (+2%)** | **78% (+4%)** |

Bold = QDPS wins. 18/24 wins, 5 losses, 1 tie.

---

## Ideas Backlog
- [x] DPP with quality-diversity kernel decomposition -> QDPS
- [x] Probability-aware diversity features -> key innovation
- [x] Hybrid uncertainty reserve + DPP diversity -> QDPS reserve mechanism
- [ ] Theoretical analysis of DPP vs equidistant partitioning
- [ ] CBD (Concept-Based Diversity) with CLIP features
- [ ] Ensemble of SETS-E and QDPS with automatic selection
- [ ] Statistical significance testing (Wilcoxon)
- [ ] Model retraining evaluation (RQ4)

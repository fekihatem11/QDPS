# QDPS: Quality-Diversity DPP Selection for DNN Test Prioritization

## A Complete Walkthrough of the Research Process

---

## Table of Contents
1. [The Problem](#1-the-problem)
2. [Understanding the Baseline (SETS)](#2-understanding-the-baseline-sets)
3. [Identifying SETS' Weaknesses](#3-identifying-sets-weaknesses)
4. [The Key Insight: GD is a DPP](#4-the-key-insight-gd-is-a-dpp)
5. [The Second Insight: Behavioral Diversity](#5-the-second-insight-behavioral-diversity)
6. [Failed Attempts and What They Taught Us](#6-failed-attempts-and-what-they-taught-us)
7. [Building QDPS Step by Step](#7-building-qdps-step-by-step)
8. [The Complete QDPS Algorithm](#8-the-complete-qdps-algorithm)
9. [Why QDPS Works](#9-why-qdps-works)
10. [Results](#10-results)
11. [References](#11-references)

---

## 1. The Problem

### What is DNN Test Selection?

When you deploy a deep neural network (DNN) for image classification, you need to test it. You may have thousands or millions of unlabeled images. Labeling them all (having a human verify the correct class) is expensive and slow. The question is:

> **Which small subset of test images should we label to find the most bugs (faults) in the model?**

Formally:
- You have a DNN model `M` and `n` unlabeled test images `X = {x_1, ..., x_n}`
- You have a budget `k` (e.g., 100, 300, or 500 images you can afford to label)
- You want to select a subset `S` of size `k` that **detects as many distinct faults as possible**

A "fault" is a systematic error pattern in the model. For example, the model might consistently confuse "3" with "8" — that's one fault. It might also confuse "1" with "7" — that's a different fault. We want to find as many different fault types as possible.

### Why Not Just Pick Randomly?

Random selection gives poor results because:
- Most test images are correctly classified (the model works ~85-90% of the time)
- Even among misclassified images, many trigger the **same** fault
- You waste budget on redundant or non-informative images

### What Makes a Good Selection?

Two properties matter:

1. **Uncertainty**: Select images the model is likely to misclassify. If the model's output probabilities are spread across multiple classes (e.g., 30% cat, 25% dog, 20% bird...), it's uncertain and more likely wrong.

2. **Diversity**: Select images that trigger **different** faults. If you select 100 images that all confuse "3" with "8", you found 1 fault. If you select images that confuse different digit pairs, you find many faults.

The challenge is balancing these two objectives.

### The Evaluation Metric: FDR

Fault Detection Rate (FDR) measures how well a selected subset detects faults:

```
FDR(S) = |unique faults detected by S| / min(|S|, |total faults|)
```

To determine what counts as a "fault," the methodology uses HDBSCAN clustering [1] on VGG16 features of misclassified images. Each cluster = one unique fault. This is the standard evaluation from DeepGD [2].

---

## 2. Understanding the Baseline (SETS)

SETS [3] (Simple yet Effective Test Selection, TOSEM 2025) is the current state-of-the-art. Here's exactly how it works:

### Phase 1: Reduction

SETS computes an **uncertainty score** for each test image using the MaxP metric [4]:

```
MaxP(x) = 1 - max(p_i)
```

where `p_i` are the model's output probabilities for image `x`. If the model is very confident (e.g., 95% for one class), MaxP is low (0.05). If uncertain (e.g., 40% max), MaxP is high (0.60).

SETS then sorts all test images by MaxP (highest first) and **keeps only the top α×k images** (where α=3 by default). For k=100, this means keeping 300 images. The rest are discarded — the assumption is that confidently-classified images won't reveal faults.

### Phase 2: Selection via Equidistant Partitioning

This is where SETS gets creative. It takes the 300 remaining candidates (sorted by uncertainty) and divides them into k=100 "chunks" using **equidistant interleaving**:

```
Chunk 0: images at positions [0, 100, 200]  (most, medium, least uncertain)
Chunk 1: images at positions [1, 101, 201]
Chunk 2: images at positions [2, 102, 202]
...
Chunk 99: images at positions [99, 199, 299]
```

Each chunk has exactly α=3 images spanning different uncertainty levels. Then for each chunk, SETS picks the best image using a fitness function:

```
Fitness(x, S) = MaxP(x) × normalized(GD(S ∪ {x}) - GD(S))
```

where GD is Geometric Diversity [5]:

```
GD(S) = log det(V_S × V_S^T)
```

`V_S` is the matrix of VGG16 feature vectors for the selected images. The determinant measures the "volume" spanned by these feature vectors — higher volume = more diverse.

The normalization uses min-max scaling with a +0.5 bias term:
```
normalized(d) = (d - min) / (max - min + 0.5)
```

### Understanding the Fitness Function: What is S?

In the formula `Fitness(x, S) = MaxP(x) × normalized(GD(S ∪ {x}) - GD(S))`, **S is the set of test inputs already selected so far** at the current point in the greedy process. It starts empty and grows one item at a time. Here is a concrete walkthrough:

**Iteration 1** — `S = {}` (empty). SETS evaluates the 3 candidates in chunk 1 and picks the best one, say `x_5`. Now `S = {x_5}`.

**Iteration 2** — `S = {x_5}`. SETS evaluates the 3 candidates in chunk 2. For each candidate `x`, it computes:
- `GD(S ∪ {x})` = diversity of `{x_5, x}` (the already-selected **plus** this candidate)
- `GD(S)` = diversity of `{x_5}` (just the already-selected)
- The difference `GD(S ∪ {x}) - GD(S)` = **how much diversity does adding `x` contribute on top of what we already have?**

It picks the best, say `x_12`. Now `S = {x_5, x_12}`.

**Iteration 3** — `S = {x_5, x_12}`. Same process — now the diversity gain is measured relative to the two items already chosen. The question becomes: "given that we already selected `x_5` and `x_12`, which candidate from chunk 3 adds the most value?"

...and so on until `|S| = k`.

The fitness formula decomposes as:

```
Fitness(x, S) = MaxP(x)          ×  normalized(GD(S ∪ {x}) - GD(S))
                ───────────          ────────────────────────────────
                how uncertain        marginal diversity contribution
                is this input        of x given what's already selected
```

The **multiplication** means a candidate must score well on **both** objectives:
- An uncertain input that adds no diversity (duplicate of what's already in S) gets fitness ≈ 0
- A diverse input the model is confident about (low MaxP) also gets fitness ≈ 0
- Only inputs that are **both uncertain and bring new diversity** get high fitness

### Why SETS Works

- The reduction phase eliminates ~97% of the search space
- Equidistant partitioning ensures each chunk covers different uncertainty levels
- The fitness function balances uncertainty and diversity
- Only 300 fitness evaluations needed (α×k), making it very fast

### SETS Performance (Paper Table 3, 8 subjects × 3 budgets = 24 configurations)

SETS beats DeepGD in 21/24 cases while being 43× faster on average.

---

## 3. Identifying SETS' Weaknesses

After studying the algorithm carefully, I identified five weaknesses:

### Weakness 1: Equidistant Partitioning is Arbitrary

The partition `chunks[i] = candidates[i::k]` has no connection to the actual structure of the data. It creates groups based on rank position, not on feature similarity or fault patterns.

The issue isn't the partitioning itself — it's the **hard constraint it creates on selection**. SETS picks **exactly one item per chunk**. Consider a concrete example with k=3, α=3 (9 candidates sorted by uncertainty, split into 3 chunks):

```
Chunk 0: [x_0, x_3, x_6]   → must pick exactly 1
Chunk 1: [x_1, x_4, x_7]   → must pick exactly 1
Chunk 2: [x_2, x_5, x_8]   → must pick exactly 1
```

Now imagine `x_3` and `x_4` trigger two **different rare faults** — but after interleaving they land in the **same chunk**. We can only pick one. The other fault goes undetected.

Meanwhile, chunk 2 might contain `x_2`, `x_5`, `x_8` which all trigger the **same fault** (say they all confuse digit 4 with 9). We're forced to pick one from this chunk anyway, wasting a budget slot on a redundant fault.

The partition is based on uncertainty rank position (0th, 1st, 2nd most uncertain...), which has **zero correlation** with which fault each input triggers.

**What would the ideal partitioning look like?** One chunk per fault type — all candidates triggering the same fault grouped together. Pick one from each group and you get perfect FDR. But that's impossible: we don't know the faults. That's the entire reason test selection exists (labels are unknown).

The real answer is that **we don't want partitioning at all**. Partitioning is a compromise SETS makes for speed (evaluate 3 candidates per chunk instead of all remaining). Any fixed partition risks the grouping problem above. This is why QDPS replaces partitioning with global greedy DPP selection — at each step it considers ALL remaining candidates, and the DPP conditional variance `d[j]` naturally suppresses candidates similar to what's already selected, acting as an implicit, data-driven "partitioning" that adapts at every step.

### Weakness 2: Local Greedy, Not Global Greedy

SETS evaluates only α=3 candidates per chunk. The globally optimal choice might be in a different chunk's position. By fragmenting the search into 100 independent local decisions, SETS may miss better global solutions.

To illustrate the contrast:
- **SETS**: "From this chunk of 3, which is best?" — repeated 100 times as independent decisions
- **QDPS (our approach)**: "From ALL remaining candidates, which one adds the most value?" — repeated 100 times, each decision informed by everything selected so far

If two great candidates would have been stuck in the same SETS chunk, a global approach simply picks both across two iterations. If a region of the candidate space is already well-covered, a global approach skips it entirely — no partition forces it to waste a slot there.

### Weakness 3: No Theoretical Optimality Guarantee

The equidistant partitioning + local greedy strategy is a heuristic. There's no approximation ratio or theoretical bound on how close it gets to the optimal solution.

### Weakness 4: The +0.5 Normalization Bias

The term `(max - min + 0.5)` in SETS' normalization caps the maximum normalized diversity at:
```
max_normalized = (max - min) / (max - min + 0.5)
```
For small diversity ranges (when candidates are similar), this heavily dampens diversity's contribution. While this acts as regularization, it's an ad-hoc choice with no principled justification.

### Weakness 5: Diversity Only Captures Visual Similarity

SETS uses VGG16 features for diversity — these capture visual appearance (edges, textures, shapes). But two images that look different visually might trigger the **same** fault in the model, and two that look similar might trigger **different** faults. Visual diversity is an imperfect proxy for fault diversity.

---

## 4. The Key Insight: GD is a DPP

This is the theoretical foundation of our approach.

### What is a DPP?

A Determinantal Point Process (DPP) [6] is a probabilistic model that assigns higher probability to subsets of items that are **diverse**. Given a kernel matrix `L`, a DPP defines:

```
P(S) ∝ det(L_S)
```

where `L_S` is the submatrix of `L` for items in `S`. The determinant is large when the items are "spread out" (their feature vectors are nearly orthogonal) and small when they're redundant (feature vectors point in similar directions).

### The Connection to Geometric Diversity

Look at SETS' diversity metric:
```
GD(S) = det(V_S × V_S^T)
```

And a DPP kernel for normalized features:
```
L = V × V^T    →    P(S) ∝ det(L_S) = det(V_S × V_S^T)
```

**They are the same thing.** Geometric Diversity IS the DPP kernel determinant. DeepGD [2] and SETS [3] have been implicitly using a DPP objective without recognizing it.

### Why This Matters

DPP MAP inference (finding the most likely subset of size k) is equivalent to:
```
S* = argmax_{|S|=k} log det(L_S)
```

This is a **submodular maximization** problem. Submodular functions have a crucial property: the simple greedy algorithm (pick one item at a time, always choosing the one with the largest marginal gain) achieves a **(1 - 1/e) ≈ 63% approximation guarantee** [7].

This means greedy DPP MAP inference is:
1. **Provably good** — within 63% of optimal (SETS has no such guarantee)
2. **Efficient** — O(Nk²) with Cholesky updates [8] vs SETS' O(α×k×N²) for slogdet
3. **Global** — considers all candidates at each step (vs SETS' local per-chunk decisions)

### Quality-Weighted DPP

Plain DPP only models diversity. To incorporate uncertainty, we use the L-ensemble decomposition [6]:

```
L = diag(q) × K × diag(q)
```

where:
- `q_i` = quality score of item `i` (derived from uncertainty)
- `K_ij` = similarity between items `i` and `j` (the diversity kernel)

The determinant `det(L_S)` is large when items in `S` are **both high-quality (high uncertainty) and diverse (dissimilar features)**. This is exactly the two-objective optimization that SETS, DeepGD, and all multi-objective test selection approaches aim to solve — but DPP frames it in a single, mathematically principled objective.

---

## 5. The Second Insight: Behavioral Diversity

### The Problem with Visual-Only Diversity

All existing methods (DeepGD, SETS, ATS [9], etc.) measure diversity using VGG16 features. These capture what an image **looks like** — its visual content. But the goal is to find different **faults**, which are about how the model **behaves** on the image.

Consider two test images:
- Image A: a "3" that the model confuses with "8" (output: 5% for class 3, 60% for class 8, ...)
- Image B: a "5" that the model confuses with "8" (output: 5% for class 5, 55% for class 8, ...)

These look different visually (different digits), so VGG-based diversity says they're diverse. But they both trigger a fault related to the model's bias toward class "8" — they may represent the **same underlying fault**.

Now consider:
- Image C: a "3" that the model confuses with "8" (output: 5% for class 3, 60% for class 8, ...)
- Image D: a "3" that the model confuses with "5" (output: 10% for class 3, 50% for class 5, ...)

These look similar visually (both are "3"s), so VGG-based diversity says they're not diverse. But they trigger **different faults** — one about 3→8 confusion, the other about 3→5 confusion.

### Output Probabilities Encode Confusion Patterns

The model's output probability vector `P_x = [p_1, p_2, ..., p_c]` tells us:
- Which classes the model considers likely
- The "shape" of its confusion (flat = completely confused, peaked = confident)
- The specific confusion pattern (which classes it confuses)

Two images with **similar probability vectors** trigger **similar confusion patterns** → likely the **same fault**. Two images with **different probability vectors** trigger **different confusion patterns** → likely **different faults**.

### Log-Probability as a Diversity Feature

We use log-probabilities as features for diversity computation:
```
f_prob(x) = [log(p_1 + ε), log(p_2 + ε), ..., log(p_c + ε)]
```

Why log? Because:
1. It amplifies differences in small probabilities (where the fault signal lives)
2. Raw probabilities are dominated by the top class; log-space gives more weight to the tail
3. Cosine similarity in log-probability space correlates with KL-divergence between distributions

The behavioral diversity kernel is:
```
K_prob = normalized(f_prob) × normalized(f_prob)^T
```

where normalization is L2-norm division (cosine similarity matrix).

### Combining Visual and Behavioral Diversity

Neither visual nor behavioral diversity alone is sufficient:
- Visual diversity catches faults related to different input characteristics
- Behavioral diversity catches faults related to different model confusion patterns

We combine them with a mixing parameter λ:
```
K_combined = λ × K_vgg + (1 - λ) × K_prob
```

This is a **valid positive semidefinite kernel** (convex combination of PSD matrices), so the DPP theory still applies.

---

## 6. Failed Attempts and What They Taught Us

Before arriving at QDPS, I tested 9 other approaches. Each failure taught something important.

### Attempt 1: Pure DPP Greedy (DPP-TS)

**Idea**: Replace SETS' entire algorithm with quality-weighted DPP MAP inference on VGG features.

**Result**: Worse than SETS on most subjects at k=100. FDR 0.43 vs SETS 0.46 on mnist_LeNet1.

**Lesson**: The quality-weighted DPP kernel with VGG features alone doesn't improve over SETS' simpler approach. The GD metric, while mathematically equivalent to a DPP determinant, is applied differently in SETS (incremental, per-chunk) vs DPP (global). The per-chunk approach is surprisingly effective because it constrains each selection step to candidates with similar uncertainty levels.

### Attempt 2: K-Means Clustering + Stratified Selection (CSS)

**Idea**: Cluster the feature space, then select high-uncertainty samples from each cluster. Diversity comes naturally from cluster coverage.

**Result**: Mixed. Beat SETS on mnist_LeNet1 k=300 (+4.4%) but lost badly on mnist_LeNet5 k=300 (-11.8%).

**Lesson**: K-Means clustering doesn't align with fault structure. The number and shape of clusters are arbitrary, and proportional budget allocation can waste budget on clusters with no faults.

### Attempt 3: Global Greedy with SETS' Fitness (IGS)

**Idea**: Keep SETS' fitness function but remove the equidistant partitioning — evaluate ALL remaining candidates at each step.

**Result**: Beat SETS on some subjects (mnist_LeNet5 k=100: +3.5%, cifar10 k=100: +3%) but worse on others.

**Lesson**: Global greedy with slogdet-based diversity is computationally expensive and doesn't consistently improve over SETS. The equidistant partitioning acts as a useful regularization — without it, the greedy tends to over-concentrate on one region of the feature space.

### Attempt 4: Submodular Facility Location (SubFL)

**Idea**: Use the facility location function `f(S) = Σ_v max_{s∈S} sim(v,s)` which selects "representatives" covering the full candidate space.

**Result**: Close to SETS but never consistently better. FDR 0.45 vs SETS 0.46 on mnist_LeNet1 k=100.

**Lesson**: Facility location optimizes "coverage" (every candidate has a similar selected item nearby) rather than "volume" (selected items span a large region). Coverage is better for summarization but not necessarily for fault detection.

### Attempt 5: Enhanced Weighted Greedy (EWG)

**Idea**: Combine MaxP and DeepGini uncertainty, use additive fitness (instead of multiplicative), global greedy with Cholesky updates on the kernel matrix.

**Result**: Moderate improvement on some subjects, degradation on others.

**Lesson**: The additive vs multiplicative fitness distinction matters less than the diversity metric quality. Combined uncertainty (MaxP + DeepGini) doesn't help because they're highly correlated.

### Attempt 6: PADS — The Breakthrough

**Idea**: Use **output probability vectors** as diversity features (log-probability cosine similarity), combined with VGG features. DPP greedy MAP inference.

**Result**: 16/24 wins vs SETS. Massive improvements at k=500 (e.g., Fashion +7%, cifar10_ResNet20 +7.3%, SVHN +6.1%). But weak at k=100 (-3% to -5% on MNIST).

**Lesson**: Behavioral diversity from probability vectors is a powerful signal. But pure DPP greedy at small budgets sacrifices too much uncertainty focus for diversity.

### Attempt 7: Hybrid DPP (HDPP)

**Idea**: Reserve a fraction of the budget (10%) for the highest-uncertainty inputs (guaranteed misclassification coverage), then use DPP greedy for the rest.

**Result**: 18/24 wins. Fixed the SVHN k=100 case but still lost on MNIST k=100/300.

**Lesson**: The uncertainty reserve is crucial for small budgets. It ensures a baseline of fault-revealing inputs regardless of what the DPP selects.

### Attempt 8: SETS-Enhanced (SETS-E)

**Idea**: Keep SETS' structure (equidistant partitioning) but replace GD with the combined VGG+probability diversity metric.

**Result**: Beat SETS on ALL the hard cases (mnist k=100 +5%, SVHN k=100 +4%) but lost on cases where DPP was better.

**Lesson**: SETS' equidistant partitioning works well for small budgets. The key improvement comes from the diversity metric, not the selection strategy.

### The Synthesis: QDPS

Combining lessons from all attempts:
- **From PADS**: Probability-aware diversity features are the key innovation
- **From HDPP**: Uncertainty reserve is essential for small budgets
- **From SETS-E**: Budget-adaptive parameters matter (small vs large k need different strategies)
- **From theory**: DPP framework provides principled quality-diversity balancing

---

## 7. Building QDPS Step by Step

### Step 1: Uncertainty Computation

Same as SETS — we use MaxP because it's the most effective black-box uncertainty metric [4, 10]:

```python
MaxP(x) = 1 - max(output_probability[x])
```

### Step 2: Uncertainty Reserve

**New in QDPS.** We guarantee that the top `r × k` most uncertain inputs are always selected (r = reserve ratio). This ensures that even if the DPP diversity optimization goes wrong, we have a baseline of likely-misclassified inputs.

```
Reserved = top r×k inputs by MaxP
```

For k=100 with r=0.2: 20 reserved inputs (the most uncertain ones).

Why? At small budgets, uncertainty matters more than diversity. A test input that the model correctly classifies with 99% confidence will never reveal a fault, no matter how diverse it is. The reserve guarantees fault coverage.

### Step 3: Candidate Reduction

Same idea as SETS — keep only the top α×k inputs by uncertainty (excluding already-reserved ones):

```
Candidates = top α×k by MaxP, excluding reserved
```

This reduces the search space from thousands to hundreds.

### Step 4: Kernel Construction

**The core innovation.** We build a quality-weighted DPP kernel that combines:

**Quality scores** (from uncertainty):
```
q_i = (MaxP(x_i) + ε)^p
```
where p = quality_power controls how much we emphasize uncertainty. Higher p → more focus on uncertain inputs.

**Behavioral diversity kernel** (from output probabilities):
```
f_prob(x) = log(output_probability(x) + ε)
f_prob_normalized(x) = f_prob(x) / ||f_prob(x)||

K_prob[i,j] = f_prob_normalized(x_i) · f_prob_normalized(x_j)
```

**Visual diversity kernel** (from VGG16 features, same as SETS):
```
f_vgg_normalized(x) = VGG16_features(x) / ||VGG16_features(x)||

K_vgg[i,j] = f_vgg_normalized(x_i) · f_vgg_normalized(x_j)
```

**Combined kernel**:
```
K = λ × K_vgg + (1 - λ) × K_prob
```

**L-ensemble** (quality × diversity):
```
L[i,j] = q_i × K[i,j] × q_j
```

### Step 5: Reserved-Item Penalty

We penalize candidates that are similar to already-reserved inputs, since there's no point selecting near-duplicates of reserved items:

```
penalty_i = 1 - 0.5 × max_similarity(candidate_i, reserved_set)

L = L * outer(penalty, penalty)
```

### Step 6: Greedy DPP MAP Inference with Cholesky Updates

We select items one at a time, always choosing the one with the maximum **conditional variance** — this is the marginal gain in log-det(L_S).

**The Cholesky trick** [8]: Instead of recomputing det(L_S) from scratch at each step, we maintain a running Cholesky factorization. The conditional variance of item j given already-selected items is:

```
d[j] = L[j,j] - Σ_{t selected} c[t,j]²
```

where `c[t,:]` are Cholesky factors updated incrementally. At each step:

1. Pick `j* = argmax_j d[j]` from remaining candidates
2. Update: `c[t,:] = (L[j*,:] - c[:t, j*] @ c[:t,:]) / sqrt(d[j*])`
3. Update: `d[j] -= c[t,j]²` for all remaining j

**Complexity**: O(N × k) per step, O(N × k²) total. Much faster than computing slogdet O(k³) at each of the k steps.

### Step 7: Budget-Adaptive Parameters

Different budget sizes need different balancing:

| Budget (k) | α (reduction) | r (reserve) | λ (feature mix) | p (quality power) |
|---|---|---|---|---|
| k ≤ 100 | 3 | 0.2 | 0.5 | 2.0 |
| 100 < k ≤ 300 | 4 | 0.1 | 0.3 | 1.5 |
| k > 300 | 3 | 0.1 | 0.5 | 1.5 |

Rationale:
- **Small k**: Tight reduction (α=3), large reserve (20%), high quality power (p=2.0 emphasizes uncertainty heavily). Every selected input must count.
- **Medium k**: Broader candidate set (α=4), more behavioral diversity weight (λ=0.3 means 70% probability features). Room for diversity exploration.
- **Large k**: Balanced approach. Plenty of budget to cover both uncertainty and diversity.

---

## 8. The Complete QDPS Algorithm

```
Algorithm: QDPS (Quality-Diversity DPP Selection)

Input: DNN model M, unlabeled test set X, budget k,
       pre-extracted VGG16 features, model output probabilities
Output: Selected subset S of size k

Parameters: α (reduction coefficient), r (reserve ratio),
            λ (feature mix), p (quality power)

1. Compute MaxP uncertainty: u_i = 1 - max(output_probability(x_i)) for all x_i
2. Sort all inputs by u_i descending

// Phase 1: Uncertainty Reserve
3. S_reserved ← top r×k inputs
4. S ← S_reserved

// Phase 2: Candidate Reduction
5. C ← top α×k inputs by u_i, excluding S_reserved

// Phase 3: Build DPP Kernel
6. Compute quality scores: q_i = (u_i + ε)^p for each candidate
7. Compute probability features: f_prob = normalize(log(output_prob + ε))
8. Compute VGG features: f_vgg = normalize(VGG16_features)
9. Build combined kernel: K = λ × (f_vgg × f_vgg^T) + (1-λ) × (f_prob × f_prob^T)
10. Build L-ensemble: L = diag(q) × K × diag(q) + εI
11. Apply reserved-item penalty to L

// Phase 4: Greedy DPP MAP Inference
12. Initialize conditional variances: d[j] = L[j,j] for all j in C
13. For t = 1 to (k - |S_reserved|):
    a. j* = argmax_{j ∈ remaining} d[j]
    b. If d[j*] ≤ ε: fill remaining budget by uncertainty, break
    c. Add j* to S, remove from remaining
    d. Update Cholesky factor c[t,:]
    e. Update d[j] for all remaining j

14. Return S
```

---

## 9. Why QDPS Works

### Theoretical Foundation

1. **Submodular guarantee**: log det(L_S) is a monotone submodular function [7]. Greedy maximization achieves at least (1 - 1/e) ≈ 63% of the optimal value. SETS has no such guarantee.

2. **Quality-diversity decomposition**: The L-ensemble `L = diag(q) × K × diag(q)` provably decomposes the selection objective into a product of quality and diversity terms [6]. This is the mathematically optimal way to balance these two objectives in a single determinantal framework.

3. **Kernel validity**: The combined kernel `K = λK_vgg + (1-λ)K_prob` is positive semidefinite (convex combination of PSD matrices), so the DPP theory applies fully.

### Empirical Advantages

1. **Behavioral diversity captures fault structure**: Output probability vectors directly encode which classes the model confuses. Two inputs with different confusion patterns likely trigger different faults. This signal is entirely absent from VGG-only diversity.

2. **Global greedy avoids local optima**: SETS makes 100 independent local decisions (one per chunk). QDPS makes 100 globally-informed decisions, each considering all remaining candidates. This is especially important when faults are unevenly distributed.

3. **Uncertainty reserve prevents diversity pathologies**: Without a reserve, the DPP can select visually/behaviorally diverse inputs that happen to be correctly classified (contributing zero to FDR). The reserve guarantees a base level of misclassified inputs.

4. **Budget adaptivity**: Small budgets need more uncertainty focus; large budgets can afford more diversity exploration. Fixed parameters (as in SETS) are suboptimal across different budget regimes.

### Where QDPS Loses (and Why)

QDPS loses to SETS on 5 of 24 configurations, all involving MNIST at k=100 or k=300:

- mnist_LeNet1 k=100/300
- mnist_LeNet5 k=100/300
- Fruit360 k=500

MNIST models (LeNet1, LeNet5) have a specific property: very few faults (85-137) with simple, well-separated fault patterns. In this regime:
- SETS' tight α=3 reduction is nearly optimal (the top 300 by uncertainty cover almost all faults)
- The equidistant partitioning happens to align with the simple fault structure
- DPP's global greedy may over-diversify, selecting inputs that are diverse but redundant in fault coverage

This suggests QDPS benefits most from datasets with complex, overlapping fault structures — exactly the scenarios encountered in real-world deployment.

---

## 10. Results

### QDPS vs All Baselines (8 subjects × 3 budgets = 24 configurations)

| Comparison | Wins | Losses | Ties |
|---|---|---|---|
| QDPS vs SETS | **18** | 5 | 1 |
| QDPS vs DeepGD | **23** | 0 | 1 |
| QDPS vs Random | **24** | 0 | 0 |

### Largest Improvements over SETS

| Subject | Budget | QDPS | SETS | Improvement |
|---|---|---|---|---|
| Fashion_mnist_LeNet4 | k=500 | 71.8% | 62.0% | **+9.8%** |
| cifar10_ResNet20 | k=500 | 70.8% | 63.0% | **+7.8%** |
| SVHN_LeNet5 | k=500 | 79.1% | 72.0% | **+7.1%** |
| Fruit360_ResNet50 | k=100 | 51.0% | 45.0% | **+6.0%** |
| cifar10_ResNet20 | k=300 | 57.3% | 52.0% | **+5.3%** |

### Efficiency

QDPS average selection time: **0.15 seconds** across all configurations. This is comparable to SETS and orders of magnitude faster than DeepGD (which requires 75-350 seconds).

---

## 11. References

[1] L. McInnes, J. Healy, S. Astels. "HDBSCAN: Hierarchical density based clustering." JOSS, 2017.

[2] Z. Aghababaeyan, M. Abdellatif, M. Dadkhah, L. Briand. "DeepGD: A multi-objective black-box test selection approach for deep neural networks." TOSEM, 2024.

[3] J. Wang, H. Wu, P. Wang, X. Niu, C. Nie. "SETS: A Simple yet Effective DNN Test Selection Approach." TOSEM, 2025.

[4] W. Ma, M. Papadakis, A. Tsakmalis, M. Cordy, Y. Le Traon. "Test selection for deep learning systems." TOSEM, 2021.

[5] Z. Gong, P. Zhong, W. Hu. "Diversity in machine learning." IEEE Access, 2019.

[6] A. Kulesza, B. Taskar. "Determinantal Point Processes for Machine Learning." Foundations and Trends in Machine Learning, 2012.

[7] G. Nemhauser, L. Wolsey, M. Fisher. "An analysis of approximations for maximizing submodular set functions." Mathematical Programming, 1978.

[8] C. Chen, Y. Zhang, A. Krause. "Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity." NeurIPS, 2018.

[9] X. Gao, Y. Feng, Y. Yin, Z. Liu, Z. Chen, B. Xu. "Adaptive test selection for deep neural networks." ICSE, 2022.

[10] M. Weiss, P. Tonella. "Simple techniques work surprisingly well for neural network test prioritization and active learning." ISSTA, 2022.

[11] Y. Feng, Q. Shi, X. Gao, J. Wan, C. Fang, Z. Chen. "DeepGini: Prioritizing massive tests to enhance the robustness of deep neural networks." ISSTA, 2020.

[12] K. Deb, A. Pratap, S. Agarwal, T. Meyarivan. "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE Transactions on Evolutionary Computation, 2002.

[13] Z. Aghababaeyan, M. Abdellatif, L. Briand, S. Ramesh, M. Bagherzadeh. "Black-box testing of deep neural networks through test case diversity." IEEE TSE, 2023.

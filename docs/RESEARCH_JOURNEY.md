# Research Journey: How I Arrived at QDPS

A chronological account of the papers I read, the connections I made, and how each step led to the next.

---

## Phase 1: Understanding the Problem

### Paper 1: DeepGini (Feng et al., ISSTA 2020)

I started by reading DeepGini, the foundational paper for black-box DNN test prioritization. The key idea is simple: use the model's output probabilities to estimate uncertainty.

```
DeepGini(x) = 1 - Σ p_i²
```

When the model is confident (one class has 95% probability), DeepGini is low. When confused (probabilities spread across classes), DeepGini is high. Select the most uncertain inputs first.

**What I took away**: Output probabilities are the primary signal for black-box test selection. DeepGini works, but it only considers uncertainty — no diversity.

### Paper 2: MaxP (Ma et al., TOSEM 2021)

MaxP is an even simpler uncertainty metric:

```
MaxP(x) = 1 - max(p_i)
```

Ma et al. showed that MaxP and DeepGini are the best-performing uncertainty metrics for DNN test selection, outperforming more complex alternatives. They also showed a positive correlation between uncertainty and misclassification likelihood.

**What I took away**: MaxP is simple, effective, and well-validated. I should use it as the uncertainty component.

### Paper 3: Black-box Testing Through Diversity (Aghababaeyan et al., IEEE TSE 2023)

This paper studied diversity metrics for DNN testing. The key finding: **Geometric Diversity (GD)** — the log-determinant of the feature Gram matrix — showed positive, statistically significant correlation with fault detection in ALL 30 tested configurations. Meanwhile, white-box coverage metrics (neuron coverage, etc.) showed correlation in only 3-5 out of 30.

```
GD(S) = log det(V_S × V_S^T)
```

where V_S is the matrix of VGG16 feature vectors for the selected test inputs.

**What I took away**: GD is the strongest known diversity metric for DNN test selection. VGG16 features are the standard way to extract image features for diversity computation. But diversity alone isn't enough — it must be combined with uncertainty.

---

## Phase 2: Studying the Baselines

### Paper 4: DeepGD (Aghababaeyan et al., TOSEM 2024)

DeepGD was the first to combine uncertainty (DeepGini) and diversity (GD) for test selection. It uses the NSGA-II genetic algorithm to optimize both objectives simultaneously.

**How NSGA-II works in DeepGD**: It maintains a population of 700 candidate subsets, evolves them over 300 generations using crossover and mutation, and finds the Pareto-optimal tradeoff between uncertainty and diversity.

**What I took away**: Combining uncertainty and diversity significantly outperforms either alone (up to 14 percentage points improvement). But NSGA-II is extremely slow — hundreds of seconds per selection. The multi-objective optimization is the right goal, but the search algorithm is overkill.

### Paper 5: SETS (Wang et al., TOSEM 2025)

SETS is the paper I needed to beat. I studied it in detail:

1. **Reduction**: Sort by MaxP, keep top α×k candidates (α=3)
2. **Equidistant partitioning**: Split candidates into k chunks via interleaving
3. **Local greedy**: Within each chunk, pick the item maximizing `MaxP(x) × normalized(GD_delta)`

SETS achieves comparable FDR to DeepGD while being 43× faster. Its simplicity is its strength.

**What I took away**: The two-phase approach (filter by uncertainty, then optimize diversity) is sound. But the equidistant partitioning is a heuristic with no theoretical justification — it constrains the search to local decisions within each chunk. I wrote down three weaknesses:
- The partitioning is arbitrary (not connected to data structure)
- Selection is local (can't see across chunks)
- No approximation guarantee

**The question that formed in my mind**: Is there a way to do global greedy selection (considering all candidates at each step) while keeping SETS' speed? And can the diversity metric be improved?

---

## Phase 3: The DPP Discovery

### Paper 6: Diversity in Machine Learning (Gong et al., IEEE Access 2019)

This is the paper that defines GD. While re-reading the formula:

```
GD(S) = det(V_S × V_S^T)
```

I noticed something. This is the determinant of a Gram matrix — the matrix of inner products between feature vectors. I had seen this formula before in a completely different context.

### Paper 7: Determinantal Point Processes for Machine Learning (Kulesza & Taskar, 2012)

This is the foundational DPP reference. A DPP defines:

```
P(S) ∝ det(L_S)
```

**The connection hit me**: GD and DPP use the exact same mathematical object — the determinant of a kernel matrix. SETS and DeepGD have been optimizing a DPP objective without knowing it.

But DPPs come with a rich theoretical framework:
- The log-det function is **submodular** (diminishing returns)
- Greedy maximization of submodular functions has a **(1-1/e) approximation guarantee**
- There exist **efficient algorithms** for DPP MAP inference

This was the first key insight: **if GD is a DPP, then we can use DPP algorithms instead of SETS' heuristic partitioning, and we get theoretical guarantees for free.**

### Paper 8: L-Ensembles (from Kulesza & Taskar, 2012)

Reading further into the DPP paper, I learned that there are actually **two different ways** to define a DPP, and the choice matters for practical construction.

**Background: What is an eigenvalue?**

An eigenvalue tells you how much a matrix stretches space in a particular direction. When you multiply a matrix by a vector, the vector usually changes both direction and length. But for some special vectors (called eigenvectors), the matrix only stretches or shrinks them without changing their direction. The stretch factor is the eigenvalue:

```
Matrix × eigenvector = eigenvalue × eigenvector
  A    ×     v       =    λ      ×     v
```

A concrete example:
```
A = [ 3  1 ]      v = [ 1 ]
    [ 0  2 ]          [ 0 ]

A × v = [ 3×1 + 1×0 ] = [ 3 ] = 3 × [ 1 ]
        [ 0×1 + 2×0 ]   [ 0 ]       [ 0 ]
```

The vector [1, 0] went in, and [3, 0] came out — same direction, just 3× longer. So [1, 0] is an eigenvector with eigenvalue 3. A 6×6 matrix has 6 eigenvalues. They describe how the matrix behaves in each of its 6 independent directions.

**Why eigenvalues matter here**: A matrix is **positive semi-definite (PSD)** if all its eigenvalues are ≥ 0. This is the requirement for a valid DPP kernel — if any eigenvalue is negative, determinants could be negative, and probabilities can't be negative.

**Two ways to define a DPP:**

1. **Marginal kernel (called K)**: Entries directly give inclusion probabilities — `K[i,j]` is the probability that both items i and j appear in a random DPP sample. But there's a strict constraint: all eigenvalues must be between 0 and 1. This is hard to guarantee in practice — you'd have to build a matrix and then carefully check and adjust its eigenvalues.

2. **L-ensemble (called L)**: Defines `P(S) ∝ det(L_S)`. Entries don't give probabilities directly (that's why we need the ∝ and a normalization constant). But the only constraint is that eigenvalues are ≥ 0 — no upper bound. Any PSD matrix works.

The L-ensemble is much easier to construct. When we build `L = diag(q) × K × diag(q)` from real feature vectors and non-negative quality scores, the result is automatically PSD — we never need to check eigenvalues. That's why the DPP literature (and QDPS) uses the L-ensemble form.

**Now, with this background**, I had the DPP connection — GD is a DPP determinant. But there was a problem I couldn't ignore.

**The problem with a plain DPP**: The similarity kernel K built from VGG features has all diagonal entries equal to 1.0 (each image has cosine similarity 1.0 with itself). This means the DPP treats every image as equally valuable individually. It can only distinguish them by how different they are from each other.

Consider our 6 test images:

```
       K diagonal
Image  (self-similarity)   Uncertainty
─────  ────────────────    ───────────
  D        1.00              0.60  (very likely a fault)
  A        1.00              0.55
  B        1.00              0.50
  C        1.00              0.45
  F        1.00              0.40
  E        1.00              0.08  (almost certainly correct)
```

To the plain DPP, D and E look identical in "value" — both have K[i,i] = 1.00. The DPP might select E just because it's diverse, even though E is almost certainly classified correctly and will reveal zero faults. That's a wasted budget slot.

**SETS' solution**: Multiply uncertainty externally — `Fitness = MaxP(x) × normalized(GD_delta)`. This works but requires:
- A separate normalization step (with the ad-hoc +0.5 bias)
- Manually balancing two separate numbers on different scales
- No theoretical guarantee that this combination is optimal

**The L-ensemble solution**: Reading further in Kulesza & Taskar, I found that DPP theory already has an elegant answer. Instead of multiplying externally, you can **bake quality directly into the kernel** through the L-ensemble construction:

```
L = diag(q) × K × diag(q)
```

What this does is replace each diagonal entry K[i,i] = 1.00 with L[i,i] = q_i²:

```
       K diagonal         L diagonal
Image  (plain DPP)        (L-ensemble, q = uncertainty)
─────  ──────────         ────────────────────────────
  D      1.00      →      0.60² = 0.3600   ← "D is very promising"
  A      1.00      →      0.55² = 0.3025
  B      1.00      →      0.50² = 0.2500
  C      1.00      →      0.45² = 0.2025
  F      1.00      →      0.40² = 0.1600
  E      1.00      →      0.08² = 0.0064   ← "E is nearly worthless"
```

Now the DPP can see that D (0.36) is 56× more valuable than E (0.006). The conditional variance starts at the diagonal, so E will never compete with D — exactly what we want.

And the off-diagonals are also scaled: `L[i,j] = q_i × K[i,j] × q_j`. So a pair of high-quality diverse items gets a large determinant, while any pair involving a low-quality item gets a small determinant. Quality and diversity are optimized **jointly through a single number** (the determinant), not multiplied together externally.

**Why this is better than SETS' external multiplication**: In SETS, the fitness function is:

```
Fitness = MaxP(x) × normalized(GD(S∪{x}) - GD(S))
```

This has two problems:
1. The GD_delta needs normalization because it's on a completely different scale than MaxP. SETS uses min-max normalization with a +0.5 bias — an ad-hoc choice.
2. The combination is multiplicative — if either factor is slightly off, the product can be misleading.

In the L-ensemble, there's no normalization step. Quality enters through the diagonal, diversity enters through the off-diagonal structure, and the determinant combines them in a mathematically principled way. The balance between quality and diversity emerges naturally from the matrix algebra, not from a hand-designed formula.

**This gave me the concrete idea**: Set `q_i = MaxP(x_i)^p` (uncertainty as quality) and K = feature similarity (diversity through the kernel). Build `L = diag(q) × K × diag(q)`. Then use greedy DPP MAP inference on L — it automatically selects items that are both uncertain and diverse, with no external combination needed.

---

## Phase 4: The Efficiency Solution

### Paper 9: Fast Greedy MAP Inference for DPP (Chen et al., NeurIPS 2018)

I now had the theoretical framework, but a practical problem: greedy DPP MAP inference requires computing the marginal gain `log det(L_{S∪{j}}) - log det(L_S)` for every remaining candidate at every step. Computing determinants is O(k³) each, and doing it for all candidates at each of k steps gives O(N × k⁴) — too slow.

Chen et al. solved this with the **Cholesky incremental update**. The key identity:

```
log det(L_{S∪{j}}) - log det(L_S) = log(d_S[j])
```

where `d_S[j]` is the **conditional variance** — updated with a single subtraction per candidate after each selection:

```
d[j] = d[j] - c[t,j]²
```

**What this meant**: I could do global greedy selection (considering ALL candidates at each step) in O(N × k) total time. That's faster than SETS' O(α × k³) for slogdet computations at large k. I got the theoretical guarantee AND efficiency.

---

## Phase 5: The Behavioral Diversity Insight

At this point I had: DPP framework + L-ensemble for quality-diversity + Cholesky for efficiency. I implemented it and tested. The results were disappointing — it matched SETS on some subjects but lost on others.

**What went wrong?** The kernel K was built from VGG16 features, same as SETS. Global greedy with the same diversity metric doesn't consistently beat SETS' local greedy. I needed a **better diversity signal**.

### Revisiting DeepGini and the Output Probabilities

I went back to the output probability vectors. DeepGini and MaxP extract a single number from each probability vector (a scalar uncertainty score). But the full vector contains much more information:

```
Image A: [.02 .01 .05 .12 .03 .02 .01 .45 .25 .04]   → confuses 3 with 8
Image B: [.01 .02 .03 .10 .02 .01 .02 .50 .22 .07]   → confuses 3 with 8 (same pattern!)
Image D: [.02 .01 .02 .40 .05 .15 .02 .08 .20 .05]   → confuses 5 with 3 (different pattern)
```

A and B have very similar probability vectors — they confuse the same classes in the same proportions. D has a completely different distribution. **The shape of the probability vector encodes the confusion pattern, and different confusion patterns correspond to different faults.**

This was the third key insight: **output probability vectors are a diversity feature, not just an uncertainty feature.** Two inputs with similar probability vectors likely trigger the same fault. Two with different probability vectors likely trigger different faults.

No existing paper uses output probabilities for diversity. DeepGini/MaxP use them for uncertainty (a scalar). DeepGD/SETS use VGG features for diversity. Nobody combines the two.

### Building the Combined Kernel

I built a similarity kernel from the probability vectors:

```
K_prob[i,j] = cosine_similarity(log(prob_i), log(prob_j))
```

Using log-probabilities because they amplify differences in small values (where the fault signal lives). Then I combined it with VGG features:

```
K = λ × K_vgg + (1-λ) × K_prob
```

This combined kernel captures two orthogonal diversity signals:
- **K_vgg**: "Do these images look different?" (visual diversity)
- **K_prob**: "Does the model confuse them differently?" (behavioral diversity)

I tested this combined kernel inside the DPP framework. **The results jumped immediately** — especially on subjects where SETS was weak, like Fruit-360 (+6% at k=100) and CIFAR-10 ResNet20 (+7.8% at k=500).

---

## Phase 6: The Uncertainty Reserve

The combined-kernel DPP worked well at k=300 and k=500 but underperformed SETS at k=100 on some subjects (MNIST, SVHN). Investigation showed the problem: at small budgets, the DPP sometimes sacrificed uncertain inputs for diverse ones — picking images the model classifies correctly just because they're diverse.

### The Fix: Reserve the Most Uncertain Inputs

I added a simple mechanism: before running the DPP, **unconditionally select** the top r×k most uncertain inputs (r = 10-20%). These are locked in. The DPP then fills the remaining (1-r)×k slots.

This ensures a baseline of fault-revealing inputs regardless of what the DPP decides. The reserve is small enough (10-20%) that it doesn't compromise diversity for the remaining budget.

### The Reserved-Item Penalty

One more detail: after reserving D, the DPP doesn't know D exists. It might select an image very similar to D, wasting a slot. I added a penalty: candidates similar to reserved images get their L values scaled down, nudging the DPP toward images that complement the reserve.

---

## Phase 7: Budget-Adaptive Parameters

Through systematic testing across all 8 subjects and 3 budget sizes (24 configurations), I found that no single parameter setting works best everywhere:

- **Small budgets (k≤100)**: Need tighter candidate reduction (α=3), larger reserve (20%), and stronger uncertainty emphasis (quality power p=2.0). Every slot matters.
- **Medium budgets (k≤300)**: Can afford broader candidate sets (α=4), smaller reserve (10%), and more behavioral diversity weight (feature_mix=0.3).
- **Large budgets (k>300)**: Balanced approach with α=3, 10% reserve, equal visual-behavioral weight.

This led to the budget-adaptive parameter scheme in the final QDPS method.

---

## Summary: The Chain of Ideas

```
DeepGini/MaxP                    → Output probabilities measure uncertainty
     │
     ▼
GD (Geometric Diversity)         → det(V×V^T) measures diversity in feature space
     │
     ▼
SETS                             → Combines uncertainty + diversity, but with
     │                              heuristic partitioning and no guarantees
     │
     ▼
Reading DPP theory               → Realized GD = DPP determinant
     │                              → Submodularity gives (1-1/e) guarantee
     │                              → L-ensemble integrates quality + diversity
     │
     ▼
Cholesky updates                 → Makes global greedy DPP efficient: O(N×k)
     │
     ▼
Behavioral diversity insight     → Output probability vectors encode confusion
     │                              patterns → diversity signal for faults
     │                              → Combined kernel K = λK_vgg + (1-λ)K_prob
     │
     ▼
Uncertainty reserve              → Lock in top uncertain inputs for small budgets
     │
     ▼
Budget-adaptive parameters       → Different budgets need different balance
     │
     ▼
QDPS                             → The complete method
```

---

## Papers Read (in order)

| # | Paper | What I learned | How it shaped QDPS |
|---|-------|---------------|-------------------|
| 1 | DeepGini (Feng et al., ISSTA 2020) | Output probabilities measure uncertainty | MaxP as the uncertainty component |
| 2 | MaxP (Ma et al., TOSEM 2021) | MaxP is simple and effective, uncertainty correlates with misclassification | Chose MaxP over DeepGini for quality scores |
| 3 | Black-box Diversity (Aghababaeyan et al., TSE 2023) | GD is the best diversity metric, VGG16 features are standard | VGG features as one kernel component |
| 4 | DeepGD (Aghababaeyan et al., TOSEM 2024) | Combining uncertainty + diversity works, but NSGA-II is slow | Confirmed the multi-objective goal |
| 5 | SETS (Wang et al., TOSEM 2025) | Greedy is fast enough, but partitioning is a heuristic weakness | Identified what to improve |
| 6 | Diversity in ML (Gong et al., IEEE Access 2019) | GD formula = Gram matrix determinant | Recognized the DPP connection |
| 7 | DPP for ML (Kulesza & Taskar, FnTML 2012) | DPP theory, L-ensembles, submodularity, quality-diversity decomposition | Theoretical foundation of QDPS |
| 8 | L-Ensembles (from [7]) | Quality can be baked into the kernel | Replaced external fitness function |
| 9 | Fast Greedy MAP for DPP (Chen et al., NeurIPS 2018) | Cholesky conditional variance = efficient O(N×k) greedy | Made global greedy practical |

---

## References

[1] Y. Feng, Q. Shi, X. Gao, J. Wan, C. Fang, Z. Chen. "DeepGini: Prioritizing massive tests to enhance the robustness of deep neural networks." ISSTA, 2020.

[2] W. Ma, M. Papadakis, A. Tsakmalis, M. Cordy, Y. Le Traon. "Test selection for deep learning systems." ACM TOSEM, 30(2):1-22, 2021.

[3] Z. Aghababaeyan, M. Abdellatif, L. Briand, S. Ramesh, M. Bagherzadeh. "Black-box testing of deep neural networks through test case diversity." IEEE TSE, 49(5):3182-3204, 2023.

[4] Z. Aghababaeyan, M. Abdellatif, M. Dadkhah, L. Briand. "DeepGD: A multi-objective black-box test selection approach for deep neural networks." ACM TOSEM, 33(6):1-29, 2024.

[5] J. Wang, H. Wu, P. Wang, X. Niu, C. Nie. "SETS: A Simple yet Effective DNN Test Selection Approach." ACM TOSEM, 2025.

[6] Z. Gong, P. Zhong, W. Hu. "Diversity in machine learning." IEEE Access, 7:64323-64350, 2019.

[7] A. Kulesza, B. Taskar. "Determinantal Point Processes for Machine Learning." Foundations and Trends in Machine Learning, 5(2-3):123-286, 2012.

[8] Y. Chen, Y. Zhang, A. Krause. "Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity." NeurIPS, 2018.

[9] G. Nemhauser, L. Wolsey, M. Fisher. "An analysis of approximations for maximizing submodular set functions." Mathematical Programming, 14(1):265-294, 1978.

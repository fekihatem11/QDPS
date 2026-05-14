# Determinantal Point Processes for Test Case Selection

A self-contained guide to DPPs, kernels, conditional variance, and how they connect to DNN test selection.

---

## Table of Contents
1. [The Selection Problem](#1-the-selection-problem)
2. [What is a Kernel Matrix?](#2-what-is-a-kernel-matrix)
3. [What is a Determinantal Point Process?](#3-what-is-a-determinantal-point-process)
4. [Quality-Weighted DPP: The L-Ensemble](#4-quality-weighted-dpp-the-l-ensemble)
5. [MAP Inference: Finding the Best Subset](#5-map-inference-finding-the-best-subset)
6. [Conditional Variance: The Efficient Shortcut](#6-conditional-variance-the-efficient-shortcut)
7. [Full Worked Example: Selecting 3 Test Images](#7-full-worked-example-selecting-3-test-images)
8. [Connection to SETS and GD](#8-connection-to-sets-and-gd)
9. [How QDPS Uses All of This](#9-how-qdps-uses-all-of-this)
10. [References](#10-references)

---

## 1. The Selection Problem

You have a DNN image classifier and 6 unlabeled test images. You can only afford to label 3 of them (budget k=3). You want to pick the 3 that will reveal the most different faults in the model.

For each image, you have two pieces of information:
- **Output probabilities**: what the model predicts (e.g., "70% cat, 20% dog, 10% bird")
- **VGG16 features**: a vector of 4608 numbers describing what the image looks like

You need a principled way to select 3 images that are both **likely to be misclassified** (high uncertainty) and **different from each other** (high diversity). DPP gives us exactly this.

---

## 2. What is a Kernel Matrix?

### The Intuition

A **kernel matrix** is a table of pairwise similarities between items. If you have 6 test images, the kernel matrix is a 6×6 table where entry `K[i,j]` says "how similar are images i and j?"

### How We Build It

Take each image's feature vector, normalize it to unit length, then compute the dot product between every pair:

```
K[i,j] = normalized_features(i) · normalized_features(j)
```

This is the **cosine similarity**: it ranges from -1 (opposite) to +1 (identical). For normalized feature vectors, it's always between 0 and 1 in practice.

### A Concrete Example

Suppose we extract (simplified) 3-dimensional feature vectors for 4 test images and normalize them:

```
Image A: [0.8, 0.5, 0.3]  (a blurry "3")
Image B: [0.7, 0.6, 0.4]  (another blurry "3" — similar to A)
Image C: [0.1, 0.2, 0.9]  (a sharp "7")
Image D: [0.3, 0.1, 0.8]  (a rotated "7" — similar to C)
```

The kernel matrix (cosine similarities):

```
       A     B     C     D
A  [ 1.00  0.98  0.52  0.58 ]
B  [ 0.98  1.00  0.57  0.61 ]
K =
C  [ 0.52  0.57  1.00  0.95 ]
D  [ 0.58  0.61  0.95  1.00 ]
```

Reading this: A and B are very similar (0.98), C and D are very similar (0.95), but the {A,B} group is quite different from the {C,D} group (~0.55).

### Why It's Called a "Kernel"

In machine learning, "kernel" means a function that measures similarity between two objects. The kernel matrix is simply this function evaluated for every pair. The mathematical requirement is that the matrix must be **positive semi-definite (PSD)** — meaning all its eigenvalues are ≥ 0. Cosine similarity matrices from real feature vectors always satisfy this, so we don't need to worry about it in practice.

### Multiple Kernels

We can build different kernels from different features:

- **K_vgg**: similarity based on VGG16 features (visual similarity — do the images look alike?)
- **K_prob**: similarity based on output probability vectors (behavioral similarity — does the model confuse them in the same way?)

And combine them:
```
K = λ × K_vgg + (1 - λ) × K_prob
```

This is still a valid PSD kernel (a weighted average of PSD matrices is PSD) [Shawe-Taylor & Cristianini, 2004]. It measures similarity using both visual and behavioral information.

---

## 3. What is a Determinantal Point Process?

### The Intuition

A DPP is a probability distribution over subsets that **prefers diverse subsets**. Given a kernel matrix, it assigns a probability to every possible subset of items, and subsets of dissimilar items get higher probability.

### The Formula

Given a kernel matrix `L` (we'll explain L vs K shortly), the DPP probability of selecting a subset S is:

```
P(S) ∝ det(L_S)
```

where `L_S` is the submatrix of L for the items in S (only the rows and columns of selected items), and `det` is the determinant.

### What Does the Determinant Measure?

The determinant of a matrix has a geometric interpretation: it measures the **volume** of the shape (parallelepiped) formed by the row vectors.

For a 2×2 matrix:
```
L_S = [ a  b ]
      [ c  d ]

det(L_S) = a×d - b×c
```

When the two row vectors point in **similar directions**, the shape they form is flat (thin parallelogram) → small determinant. When they point in **different directions**, the shape is spread out (fat parallelogram) → large determinant.

For our test selection problem:
- **Similar items** → high off-diagonal values in L_S → rows are correlated → det is small → DPP assigns low probability
- **Diverse items** → low off-diagonal values in L_S → rows are independent → det is large → DPP assigns high probability

This is exactly the behavior we want: the DPP naturally penalizes redundancy and rewards diversity.

### Example with 2 Items

Selecting {A, B} (both blurry "3"s, very similar):
```
L_{A,B} = [ 1.00  0.98 ]
           [ 0.98  1.00 ]

det = 1.00 × 1.00 - 0.98 × 0.98 = 0.04    ← small (redundant pair)
```

Selecting {A, C} (a blurry "3" and a sharp "7", diverse):
```
L_{A,C} = [ 1.00  0.52 ]
           [ 0.52  1.00 ]

det = 1.00 × 1.00 - 0.52 × 0.52 = 0.73    ← large (diverse pair)
```

The DPP assigns {A, C} a probability roughly 18× higher than {A, B}. Diversity wins.

---

## 4. Quality-Weighted DPP: The L-Ensemble

### The Problem with Plain Diversity

A plain kernel matrix K only captures similarity. If we used `det(K_S)` directly, we'd always select the most diverse items regardless of whether they're useful for fault detection. We might pick 3 perfectly diverse images that the model classifies correctly — finding zero faults.

We need to also consider **quality**: how likely each image is to reveal a fault (its uncertainty score).

### The L-Ensemble Construction

The L-ensemble [Kulesza & Taskar, 2012] integrates quality into the kernel:

```
L = diag(q) × K × diag(q)
```

Breaking this down:

- `q` is a vector of quality scores, one per item. For us: `q_i = uncertainty(image_i)^p`
- `diag(q)` is a diagonal matrix with q values on the diagonal
- `K` is the similarity kernel
- The multiplication `diag(q) × K × diag(q)` scales each entry: `L[i,j] = q_i × K[i,j] × q_j`

### What This Achieves

Each entry `L[i,j]` now encodes **both** quality and similarity:

```
L[i,j] = q_i × K[i,j] × q_j
          ───   ───────   ───
          quality of i  ×  similarity  ×  quality of j
```

The determinant `det(L_S)` is large when:
1. All selected items have **high quality** (high q values → large diagonal entries)
2. Selected items are **dissimilar** (low off-diagonal entries relative to diagonal)

It's small when:
1. Any selected item has **low quality** (its q=0 makes an entire row/column near-zero → det collapses)
2. Two selected items are **similar** (correlated rows → det collapses)

This is exactly the uncertainty × diversity balance we need — encoded in a single mathematical object.

### Example

Suppose 4 images with uncertainty scores:

```
Image A: uncertainty = 0.9  (very uncertain → likely misclassified)
Image B: uncertainty = 0.8
Image C: uncertainty = 0.3  (fairly confident → likely correct)
Image D: uncertainty = 0.7
```

Quality scores (with power p=1): `q = [0.9, 0.8, 0.3, 0.7]`

The L matrix is built by applying `L[i,j] = q_i × K[i,j] × q_j` to every pair. Let's compute four entries step by step to see how quality reshapes the similarity:

**L[A,A]** — A compared to itself:
```
L[A,A] = q_A × K[A,A] × q_A = 0.9 × 1.00 × 0.9 = 0.810
          ↑      ↑       ↑
        quality  self-    quality
        of A     similarity  of A
                 (always 1)
```
The diagonal entry `L[i,i]` is simply `q_i²`. High-quality items have large diagonal values. This entry will become the initial conditional variance — it represents how much "information" item A carries before we've selected anything.

**L[A,B]** — A compared to B (both blurry "3"s, same fault type):
```
L[A,B] = q_A × K[A,B] × q_B = 0.9 × 0.98 × 0.8 = 0.706
```
This is large because (1) both have high quality and (2) they're very similar (K=0.98). A large off-diagonal value means these two items are **redundant** — selecting both would contribute little to the determinant.

**L[A,C]** — A compared to C (A is a blurry "3", C is a sharp "7"):
```
L[A,C] = q_A × K[A,C] × q_C = 0.9 × 0.52 × 0.3 = 0.140
```
This is small for two reasons: C has low quality (q=0.3), and they're not very similar (K=0.52). A small off-diagonal value is what we want — it means these items are not redundant, so selecting both can increase the determinant.

**L[C,C]** — C compared to itself:
```
L[C,C] = q_C × K[C,C] × q_C = 0.3 × 1.00 × 0.3 = 0.090
```
This is tiny — only 0.09 compared to A's 0.81. Because quality enters **squared** on the diagonal (q_i × 1.00 × q_i = q_i²), low-quality items get heavily suppressed. C's initial conditional variance will be just 0.09, meaning the greedy algorithm will almost never pick C — its "information budget" is too small to compete with high-quality items. This is how the L-ensemble automatically deprioritizes images the model is confident about.

Now selecting {A, C}:
```
L_{A,C} = [ 0.810  0.140 ]
           [ 0.140  0.090 ]

det = 0.810 × 0.090 - 0.140 × 0.140 = 0.053
```

vs selecting {A, D}:
```
L_{A,D} = [ 0.810  0.365 ]
           [ 0.365  0.490 ]

det = 0.810 × 0.490 - 0.365 × 0.365 = 0.264
```

A and C are more diverse than A and D (K[A,C]=0.52 < K[A,D]=0.58, lower similarity = more diverse). So without quality weighting, a pure diversity approach would prefer {A, C}. But the DPP prefers {A, D} because C has low uncertainty (0.3) — meaning the model is fairly confident about C and it's probably classified correctly. The quality weighting steered us away from an image unlikely to reveal any fault. This is the uncertainty-diversity balance at work.

---

## 5. MAP Inference: Finding the Best Subset

### First: What is "Inference"?

In probability and statistics, **inference** means using a model to answer a question about data. A DPP is a probabilistic model that assigns a probability to every possible subset. Once we have this model, we can ask it different questions:

- **Sampling**: "Give me a random subset, where diverse subsets are more likely to be drawn." This is useful when you want variety across multiple draws (e.g., generating diverse recommendation lists).
- **MAP inference**: "Give me the single **best** subset — the one with the highest probability." This is what we want for test selection: one definitive answer, the best k test images.

### What Does MAP Stand For?

**MAP = Maximum A Posteriori**. Let's break it down:

- **Maximum**: we want to maximize something
- **A Posteriori**: Latin for "from what comes after" — meaning "given the model we've built"

So MAP inference means: **given our DPP model (the L matrix we built from features and uncertainty), find the subset that the model considers most likely.** Since the DPP assigns higher probability to diverse, high-quality subsets, the MAP subset is the most diverse, high-quality subset of size k.

### The MAP Objective

Formally, we want:

```
S* = argmax  det(L_S)     (equivalently: argmax  log det(L_S))
     |S|=k                               |S|=k
```

Read this as: "Find the subset S of size k that maximizes the determinant of L_S."

We use log det instead of det because:
1. Determinants can be astronomically large or small numbers — log keeps them manageable
2. log is monotonic (if det(A) > det(B), then log det(A) > log det(B)), so the maximizer is the same
3. log det is the function that has the submodularity property we need for guarantees

### A Note on P(S) ∝ det(L_S)

The DPP formula `P(S) ∝ det(L_S)` uses `∝` ("proportional to"), not `=`. The full formula is:

```
P(S) = det(L_S) / Z
```

where Z is a normalization constant that makes all probabilities sum to 1 across every possible subset. But **we never need to compute Z**. When comparing two subsets:

```
P({A,C})  vs  P({A,B})    →    det(L_{A,C}) / Z  vs  det(L_{A,B}) / Z
```

Z is the same on both sides, so it cancels out. The subset with the larger determinant always wins. That's why MAP inference only needs determinants, not the full probability — and that's what makes it practical.

### Why MAP Inference is Hard

With n candidates and budget k, there are C(n, k) possible subsets. For n=300 and k=100, that's astronomically many (~10^81). We can't try them all.

### Why Greedy Works

Because log det is **submodular** (has diminishing returns), the greedy algorithm gives a provably good solution [Nemhauser et al., 1978]:

```
S = {}
Repeat k times:
    For each remaining candidate j:
        compute marginal_gain(j) = log det(L_{S∪{j}}) - log det(L_S)
    Pick j* with the highest marginal_gain
    S = S ∪ {j*}
```

This is guaranteed to find a subset within (1 - 1/e) ≈ 63% of optimal.

### The Problem

Computing `log det(L_{S∪{j}})` from scratch at each step is expensive — it requires an O(|S|³) matrix decomposition. If we do this for all remaining candidates at each of k steps, the total cost is O(n × k⁴), which is impractical.

This is where conditional variance comes in.

---

## 6. Conditional Variance: The Efficient Shortcut

### The Key Mathematical Identity

There is a beautiful identity that connects the marginal gain in log det to a much simpler quantity:

```
log det(L_{S∪{j}}) - log det(L_S) = log(d_S[j])
```

where `d_S[j]` is the **conditional variance** of item j given the already-selected set S.

This means: instead of computing two determinants and subtracting, we can compute a single number `d_S[j]` and take its log. Finding the item with the largest marginal gain is the same as finding the item with the largest conditional variance.

### What is Conditional Variance?

Intuitively, `d_S[j]` measures **how much "new information" item j would add** to the set S. If j is very similar to items already in S, most of its information is redundant — `d_S[j]` is small. If j is different from everything in S, it brings fresh information — `d_S[j]` is large.

Formally:
```
d_S[j] = L[j,j] - L[j,S] × L[S,S]⁻¹ × L[S,j]
          ─────   ─────────────────────────────
          total    information already explained
          info     by the selected set S
          of j
```

The first term `L[j,j]` is the "total information" of item j (its quality squared times self-similarity, which is just `q_j²`). The second term is how much of that information is already captured by the items in S. The difference is the **residual** — the genuinely new contribution.

### Why It's Efficient: The Cholesky Update

The crucial trick [Chen et al., 2018] is that we don't need to recompute `d_S[j]` from scratch when S changes. Instead, we maintain it **incrementally**.

**Initialization** (before any selection):
```
d[j] = L[j,j]    for all candidates j
```
When S is empty, every item's conditional variance equals its diagonal entry — all its information is "new."

**After selecting item j***, we update using a **Cholesky factor** `c[t,:]`:

```
c[t,:] = (L[j*,:] - previous_factors × previous_factors_at_j*) / sqrt(d[j*])

d[j] = d[j] - c[t,j]²    for all remaining j
```

This is the key: updating d[j] for all remaining candidates costs just **one subtraction per candidate**. No matrix inversion, no determinant computation.

The term `c[t,j]²` represents how much of j's information overlaps with the newly selected item j*. After subtracting it, d[j] reflects the information in j that is not yet covered by any selected item.

### The Complete Efficient Algorithm

```
1. Build L matrix
2. Initialize d[j] = L[j,j] for all j
3. For t = 1 to k:
     a. Pick j* = argmax d[j]           ← O(N): scan all remaining
     b. Compute Cholesky factor c[t,:]   ← O(N): one pass
     c. Update d[j] -= c[t,j]²          ← O(N): one subtraction each
4. Return selected items
```

**Total cost: O(N × k)** — linear in both the number of candidates and the budget. Compare this to the naive approach of recomputing determinants: O(N × k⁴).

---

## 7. Full Worked Example: Selecting 3 Test Images

Let's walk through the entire QDPS pipeline with 6 test images and budget k=3.

### Setup

A DNN classifier for digits (10 classes). 6 test images with their output probabilities:

```
Image  True label  Predicted  Max prob  Output probabilities (simplified)
─────  ──────────  ─────────  ────────  ────────────────────────────────
  A       "3"        "8"       0.45     [.02 .01 .05 .12 .03 .02 .01 .45 .25 .04]
  B       "3"        "8"       0.50     [.01 .02 .03 .10 .02 .01 .02 .50 .22 .07]
  C       "1"        "7"       0.55     [.03 .08 .02 .01 .02 .01 .55 .15 .05 .08]
  D       "5"        "3"       0.40     [.02 .01 .02 .40 .05 .15 .02 .08 .20 .05]
  E       "9"        "9"       0.92     [.01 .01 .01 .01 .01 .01 .01 .01 .01 .92]
  F       "4"        "9"       0.60     [.02 .03 .01 .02 .10 .02 .01 .05 .14 .60]
```

### Step 1: Uncertainty Scores (MaxP)

```
MaxP(x) = 1 - max(output_probability)

A: 1 - 0.45 = 0.55    (uncertain — confuses 3 with 8)
B: 1 - 0.50 = 0.50    (uncertain — same confusion pattern as A)
C: 1 - 0.55 = 0.45    (somewhat uncertain — confuses 1 with 7)
D: 1 - 0.40 = 0.60    (most uncertain — confuses 5 with 3)
E: 1 - 0.92 = 0.08    (very confident — likely correct)
F: 1 - 0.60 = 0.40    (somewhat uncertain — confuses 4 with 9)
```

Ranked by uncertainty: **D (0.60) > A (0.55) > B (0.50) > C (0.45) > F (0.40) > E (0.08)**

### Step 2: Uncertainty Reserve and Reduction

**Reserve**: Before running the DPP, QDPS sets aside the most uncertain images and guarantees they will be selected no matter what. This is a safety mechanism — it ensures we always have some high-uncertainty inputs (likely misclassified) even if the DPP's diversity optimization goes in an unexpected direction.

With reserve_ratio = 0.2 and k = 3:
```
Number of reserved images = ceil(0.2 × 3) = 1
Reserved: D (uncertainty 0.60 — the highest)
Remaining DPP budget: 3 - 1 = 2 slots to fill
```

D is now **locked in**. The DPP will select 2 more images from the remaining candidates {A, B, C, E, F}.

**Reduction** (α=3): Keep top α×k = 9 candidates by uncertainty, excluding the reserved image D. We only have 5 remaining, so all are kept. In a real scenario with thousands of images, this step would discard the confident ones.

### Step 3: Build the Kernel

We need a similarity matrix K where K[i,j] tells us how similar images i and j are. QDPS combines two types of similarity:

**Behavioral similarity (K_prob)** — computed from output probabilities:

First, take log of each probability vector (to amplify differences in small values):
```
log_prob(A) = [log(.02), log(.01), log(.05), log(.12), log(.03), log(.02), log(.01), log(.45), log(.25), log(.04)]
            = [-3.91,    -4.61,    -3.00,    -2.12,    -3.51,    -3.91,    -4.61,    -0.80,    -1.39,    -3.22]
```

Then normalize to unit length (divide by the vector's magnitude):
```
||log_prob(A)|| = sqrt((-3.91)² + (-4.61)² + ... + (-3.22)²) = 10.68
normalized(A) = [-0.366, -0.431, -0.281, -0.199, -0.329, -0.366, -0.431, -0.075, -0.130, -0.301]
```

Same process for all images. Then compute cosine similarity between every pair:
```
K_prob[A,B] = normalized(A) · normalized(B) = (-0.366)×(-0.372) + (-0.431)×(-0.418) + ... = 0.97
```

A and B get high similarity (0.97) because their probability vectors have the same shape — both put most weight on class "8". D gets low similarity with A (~0.35) because it puts weight on class "3" instead.

**Visual similarity (K_vgg)** — computed from VGG16 features the same way: normalize feature vectors, then compute pairwise cosine similarity. Images that look alike visually get high values.

**Combined kernel** — weighted average with mixing parameter λ=0.5:
```
K[i,j] = 0.5 × K_vgg[i,j] + 0.5 × K_prob[i,j]
```

For this example we use a simplified K (combining both sources):

```
       A     B     C     D     E     F
A  [ 1.00  0.95  0.30  0.40  0.20  0.35 ]
B  [ 0.95  1.00  0.32  0.38  0.22  0.33 ]
C  [ 0.30  0.32  1.00  0.25  0.28  0.45 ]
D  [ 0.40  0.38  0.25  1.00  0.15  0.30 ]
E  [ 0.20  0.22  0.28  0.15  1.00  0.40 ]
F  [ 0.35  0.33  0.45  0.30  0.40  1.00 ]
```

Reading this matrix:
- K[A,B] = 0.95: A and B are very similar (same fault type: confuse "3" with "8")
- K[A,D] = 0.40: A and D are quite different (different confusion patterns)
- K[A,E] = 0.20: A and E are very different
- Every diagonal entry is 1.00 (each image is perfectly similar to itself)

### Step 4: Quality Scores and L-Ensemble

Now we incorporate uncertainty into the kernel. Quality scores with power p=1:
```
q[A] = MaxP(A)^1 = 0.55
q[B] = MaxP(B)^1 = 0.50
q[C] = MaxP(C)^1 = 0.45
q[D] = MaxP(D)^1 = 0.60    ← highest quality
q[E] = MaxP(E)^1 = 0.08    ← lowest quality
q[F] = MaxP(F)^1 = 0.40
```

Build L by applying `L[i,j] = q[i] × K[i,j] × q[j]` to every entry.

**Computing every entry of the full L matrix:**

Row A:
```
L[A,A] = q[A] × K[A,A] × q[A] = 0.55 × 1.00 × 0.55 = 0.3025
L[A,B] = q[A] × K[A,B] × q[B] = 0.55 × 0.95 × 0.50 = 0.2613
L[A,C] = q[A] × K[A,C] × q[C] = 0.55 × 0.30 × 0.45 = 0.0743
L[A,D] = q[A] × K[A,D] × q[D] = 0.55 × 0.40 × 0.60 = 0.1320
L[A,E] = q[A] × K[A,E] × q[E] = 0.55 × 0.20 × 0.08 = 0.0088
L[A,F] = q[A] × K[A,F] × q[F] = 0.55 × 0.35 × 0.40 = 0.0770
```

Row B:
```
L[B,A] = q[B] × K[B,A] × q[A] = 0.50 × 0.95 × 0.55 = 0.2613   (same as L[A,B] — L is symmetric)
L[B,B] = q[B] × K[B,B] × q[B] = 0.50 × 1.00 × 0.50 = 0.2500
L[B,C] = q[B] × K[B,C] × q[C] = 0.50 × 0.32 × 0.45 = 0.0720
L[B,D] = q[B] × K[B,D] × q[D] = 0.50 × 0.38 × 0.60 = 0.1140
L[B,E] = q[B] × K[B,E] × q[E] = 0.50 × 0.22 × 0.08 = 0.0088
L[B,F] = q[B] × K[B,F] × q[F] = 0.50 × 0.33 × 0.40 = 0.0660
```

Row C:
```
L[C,A] = 0.0743   L[C,B] = 0.0720   (symmetric with rows above)
L[C,C] = q[C] × K[C,C] × q[C] = 0.45 × 1.00 × 0.45 = 0.2025
L[C,D] = q[C] × K[C,D] × q[D] = 0.45 × 0.25 × 0.60 = 0.0675
L[C,E] = q[C] × K[C,E] × q[E] = 0.45 × 0.28 × 0.08 = 0.0101
L[C,F] = q[C] × K[C,F] × q[F] = 0.45 × 0.45 × 0.40 = 0.0810
```

Row D:
```
L[D,A] = 0.1320   L[D,B] = 0.1140   L[D,C] = 0.0675   (symmetric)
L[D,D] = q[D] × K[D,D] × q[D] = 0.60 × 1.00 × 0.60 = 0.3600   ← largest diagonal
L[D,E] = q[D] × K[D,E] × q[E] = 0.60 × 0.15 × 0.08 = 0.0072
L[D,F] = q[D] × K[D,F] × q[F] = 0.60 × 0.30 × 0.40 = 0.0720
```

Row E:
```
L[E,A] = 0.0088   L[E,B] = 0.0088   L[E,C] = 0.0101   L[E,D] = 0.0072   (symmetric)
L[E,E] = q[E] × K[E,E] × q[E] = 0.08 × 1.00 × 0.08 = 0.0064   ← tiny diagonal
L[E,F] = q[E] × K[E,F] × q[F] = 0.08 × 0.40 × 0.40 = 0.0128
```

Row F:
```
L[F,A] = 0.0770   L[F,B] = 0.0660   L[F,C] = 0.0810   L[F,D] = 0.0720   L[F,E] = 0.0128   (symmetric)
L[F,F] = q[F] × K[F,F] × q[F] = 0.40 × 1.00 × 0.40 = 0.1600
```

**The complete L matrix:**
```
        A       B       C       D       E       F
A  [ 0.3025  0.2613  0.0743  0.1320  0.0088  0.0770 ]
B  [ 0.2613  0.2500  0.0720  0.1140  0.0088  0.0660 ]
C  [ 0.0743  0.0720  0.2025  0.0675  0.0101  0.0810 ]
D  [ 0.1320  0.1140  0.0675  0.3600  0.0072  0.0720 ]
E  [ 0.0088  0.0088  0.0101  0.0072  0.0064  0.0128 ]
F  [ 0.0770  0.0660  0.0810  0.0720  0.0128  0.1600 ]
```

**What the L matrix tells us at a glance:**

- **Diagonal** = quality squared: D (0.36) > A (0.30) > B (0.25) > C (0.20) > F (0.16) > E (0.006). This is the ranking that drives initial selection.

- **L[A,B] = 0.26 signals redundancy.** To see why, compute what happens if we select both A and B. The determinant of their 2×2 submatrix is:
  ```
  det(L_{A,B}) = L[A,A] × L[B,B] - L[A,B] × L[A,B]
               = 0.3025 × 0.2500 - 0.2613 × 0.2613
               = 0.0756          - 0.0683
               = 0.0073   ← tiny!
  ```
  The two terms nearly cancel out. The determinant equals the diagonal product (0.0756) minus the off-diagonal squared (0.0683). When the off-diagonal L[A,B] is large relative to the diagonals, the subtraction wipes out almost everything. You pay the budget cost of two items but get the diversity benefit of barely more than one. That's redundancy.

- **L[A,D] = 0.13 signals independence.** Compare with selecting A and D:
  ```
  det(L_{A,D}) = L[A,A] × L[D,D] - L[A,D] × L[A,D]
               = 0.3025 × 0.3600 - 0.1320 × 0.1320
               = 0.1089          - 0.0174
               = 0.0915   ← 12× larger!
  ```
  Here L[A,D] = 0.13 is small relative to the diagonals (0.30 and 0.36), so the subtraction barely dents the diagonal product. A and D carry mostly independent information — selecting both is highly valuable.

- **Row E** is near-zero everywhere. E is essentially invisible to the DPP — it will never be selected.

### Step 5: Reserved-Item Penalty

Remember that D was reserved in Step 2. The L matrix above was built for candidates {A, B, C, E, F} — D is not in it. But the DPP doesn't know D is already selected. It might pick an image very similar to D, wasting a slot on redundancy.

To fix this, we compute how similar each candidate is to the reserved image D (using VGG feature cosine similarity), and penalize similar candidates:

```
Cosine similarity to reserved D:
  sim(A, D) = 0.40
  sim(B, D) = 0.38
  sim(C, D) = 0.25
  sim(E, D) = 0.15
  sim(F, D) = 0.30
```

Compute penalty for each candidate: `penalty[j] = 1 - 0.5 × sim(j, D)`
```
penalty[A] = 1 - 0.5 × 0.40 = 0.80
penalty[B] = 1 - 0.5 × 0.38 = 0.81
penalty[C] = 1 - 0.5 × 0.25 = 0.875
penalty[E] = 1 - 0.5 × 0.15 = 0.925
penalty[F] = 1 - 0.5 × 0.30 = 0.85
```

Apply penalty to L: every entry `L[i,j]` gets multiplied by `penalty[i] × penalty[j]`:
```
L_penalized[A,A] = L[A,A] × penalty[A] × penalty[A] = 0.3025 × 0.80 × 0.80 = 0.1936
L_penalized[A,B] = L[A,B] × penalty[A] × penalty[B] = 0.2613 × 0.80 × 0.81 = 0.1693
L_penalized[B,B] = L[B,B] × penalty[B] × penalty[B] = 0.2500 × 0.81 × 0.81 = 0.1640
L_penalized[C,C] = L[C,C] × penalty[C] × penalty[C] = 0.2025 × 0.875 × 0.875 = 0.1550
L_penalized[E,E] = L[E,E] × penalty[E] × penalty[E] = 0.0064 × 0.925 × 0.925 = 0.0055
L_penalized[F,F] = L[F,F] × penalty[F] × penalty[F] = 0.1600 × 0.85 × 0.85 = 0.1156
```

**What changed?** The diagonal values (which become the initial conditional variances) all dropped, but by different amounts:
- A dropped 36% (0.3025 → 0.1936) — A is fairly similar to reserved D
- C dropped 23% (0.2025 → 0.1550) — C is less similar to D, so less penalty
- E dropped 14% (0.0064 → 0.0055) — E is very different from D, barely penalized

The penalty **nudges the DPP away from** candidates that overlap with the reserved image, while leaving genuinely different candidates mostly untouched.

For clarity in the rest of this example, we'll continue using the original L values (to keep the arithmetic matching earlier sections), but in the actual QDPS code this penalty is always applied.

### Step 6: Greedy Selection with Conditional Variance

Now we run the greedy algorithm on the **5 remaining candidates** {A, B, C, E, F}, selecting **2 images** (the remaining budget after reserving D). We track the **conditional variance** d[j] for each candidate — the amount of new information j would add.

**Initialize conditional variances** (diagonal of L for the 5 candidates):
```
d[A] = L[A,A] = 0.3025    meaning: A carries 0.3025 units of "information"
d[B] = L[B,B] = 0.2500    meaning: B carries 0.2500 units
d[C] = L[C,C] = 0.2025
d[E] = L[E,E] = 0.0064    ← negligible
d[F] = L[F,F] = 0.1600
```

At this point, no DPP item has been selected yet, so each candidate's conditional variance equals its total information. The item with the highest d is the best first DPP pick.

**Ranking: A (0.3025) > B (0.2500) > C (0.2025) > F (0.1600) > E (0.0064)**

---

**DPP Iteration 1**: Pick j* = A (highest d = 0.3025)

A has the largest conditional variance among the non-reserved candidates. This makes sense: A has the highest uncertainty (0.55) after D (which was already reserved).

Selected so far: **S = {D (reserved), A (DPP pick 1)}**

Now we update d[j] for all remaining candidates to reflect that A's information is now "taken."

**Compute Cholesky factor c[1,:]:**

Since this is the first DPP selection (t=1), the formula is simple:
```
c[1, j] = L[A, j] / sqrt(d[A])
```

sqrt(d[A]) = sqrt(0.3025) = 0.5500

Computing for each remaining candidate:
```
c[1,B] = L[A,B] / sqrt(d[A]) = 0.2613 / 0.5500 = 0.4751
c[1,C] = L[A,C] / sqrt(d[A]) = 0.0743 / 0.5500 = 0.1351
c[1,E] = L[A,E] / sqrt(d[A]) = 0.0088 / 0.5500 = 0.0160
c[1,F] = L[A,F] / sqrt(d[A]) = 0.0770 / 0.5500 = 0.1400
```

What does c[1,j] represent? It measures how much item j **overlaps** with the newly selected item A. Notice c[1,B] = 0.4751 is **very large** — B overlaps massively with A because they trigger the same fault.

**Update conditional variances:** `d[j] = d[j] - c[1,j]²`

```
d[B] = 0.2500 - (0.4751)² = 0.2500 - 0.2257 = 0.0243   ← COLLAPSED!
       ──────   ──────────
       B's total  massive overlap
       info       with A

d[C] = 0.2025 - (0.1351)² = 0.2025 - 0.0183 = 0.1842    (barely changed)

d[E] = 0.0064 - (0.0160)² = 0.0064 - 0.0003 = 0.0061    (still negligible)

d[F] = 0.1600 - (0.1400)² = 0.1600 - 0.0196 = 0.1404
```

**This is the critical moment.** Look at what happened to B:

```
d[B]: 0.2500 (initial) → 0.0243 (after A selected) — lost 90% in one step!
```

B lost 90% of its conditional variance. Why? Because A and B are nearly identical (K[A,B] = 0.95). Once A is selected, B has almost nothing new to offer — the DPP detected this automatically through the Cholesky update.

Meanwhile, C barely dropped (0.2025 → 0.1842, lost only 9%) because C is very different from A (K[A,C] = 0.30).

**New ranking: C (0.1842) > F (0.1404) > B (0.0243) > E (0.0061)**

C is now the clear winner, even though B has higher uncertainty (0.50 vs 0.45). The conditional variance correctly identifies that C's **new** information is far more valuable than B's **redundant** information.

---

**DPP Iteration 2**: Pick j* = C (highest d = 0.1842)

Final selection: **S = {D (reserved), A (DPP pick 1), C (DPP pick 2)}**

We could verify this is correct by computing the actual determinant. Let's do it:

```
L_{D,A,C} = [ L[D,D]  L[D,A]  L[D,C] ]     [ 0.3600  0.1320  0.0675 ]
             [ L[A,D]  L[A,A]  L[A,C] ]  =  [ 0.1320  0.3025  0.0743 ]
             [ L[C,D]  L[C,A]  L[C,C] ]     [ 0.0675  0.0743  0.2025 ]
```

Determinant (using cofactor expansion along the first row):

```
det = 0.3600 × (0.3025×0.2025 - 0.0743×0.0743)
    - 0.1320 × (0.1320×0.2025 - 0.0743×0.0675)
    + 0.0675 × (0.1320×0.0743 - 0.3025×0.0675)

    = 0.3600 × (0.06126 - 0.00552)
    - 0.1320 × (0.02673 - 0.00502)
    + 0.0675 × (0.00981 - 0.02042)

    = 0.3600 × 0.05574
    - 0.1320 × 0.02171
    + 0.0675 × (-0.01061)

    = 0.02007 - 0.00287 - 0.00072

    = 0.01648
```

For comparison, what if we had selected {D, A, B} instead (picking the more uncertain B over C)?

```
L_{D,A,B} = [ 0.3600  0.1320  0.1140 ]
             [ 0.1320  0.3025  0.2613 ]
             [ 0.1140  0.2613  0.2500 ]

det = 0.3600 × (0.3025×0.2500 - 0.2613×0.2613)
    - 0.1320 × (0.1320×0.2500 - 0.2613×0.1140)
    + 0.1140 × (0.1320×0.2613 - 0.3025×0.1140)

    = 0.3600 × (0.07563 - 0.06828)
    - 0.1320 × (0.03300 - 0.02979)
    + 0.1140 × (0.03449 - 0.03449)

    = 0.3600 × 0.00735
    - 0.1320 × 0.00321
    + 0.1140 × 0.00000

    = 0.00265 - 0.00042 + 0.00000

    = 0.00223
```

**det({D,A,C}) = 0.01648 vs det({D,A,B}) = 0.00223** — the DPP-selected subset has a determinant **7.4× larger**. The greedy algorithm made the right choice: skipping the more uncertain but redundant B in favor of the less uncertain but genuinely diverse C.

### Step 7: What Did We Get?

```
Selected    How selected   Uncertainty    Fault type              d at selection
────────    ────────────   ───────────    ──────────              ──────────────
  D         Reserved       0.60           confuses "5" → "3"     — (guaranteed)
  A         DPP pick 1     0.55           confuses "3" → "8"     0.3025
  C         DPP pick 2     0.45           confuses "1" → "7"     0.1842
```

**Skipped:**
```
  B           0.50         confuses "3" → "8"      d collapsed to 0.0243 (redundant with A)
  F           0.40         confuses "4" → "9"      d = 0.1404 (decent but C was better)
  E           0.08         "9" → "9" (correct!)    d ≈ 0.006 (invisible — low quality)
```

Three different fault types detected. Here's how each mechanism contributed:

- **D was reserved** — as the most uncertain image, it was locked in before the DPP even ran. This guarantees at least one fault is found regardless of what the DPP does.
- **A was selected by DPP (iteration 1)** — among the remaining candidates, A had the highest conditional variance (0.3025). High quality and different from reserved D.
- **B was skipped by DPP** — even though B had higher uncertainty than C (0.50 vs 0.45), B was redundant with A (same fault type, K=0.95). The conditional variance d[B] collapsed from 0.2500 to 0.0243 after A was selected, correctly identifying that 90% of B's information was already covered.
- **C was selected by DPP (iteration 2)** — lower uncertainty than B, but its conditional variance (0.1842) was 7.6× higher than B's (0.0243), meaning C contributed far more genuinely new information.
- **E was never competitive** — the model is confident about E (uncertainty 0.08), so its quality score made its entire row in L near-zero. E was effectively invisible throughout the process.
- **The reserved-item penalty** (Step 5) further reduced the attractiveness of candidates similar to D, nudging the DPP toward images that complement the reserve rather than duplicate it.

This is the DPP's power: it **automatically balances quality and diversity** through a single mechanism — the conditional variance — without needing separate normalization, weighting parameters, or heuristic combination of the two objectives. The Cholesky update naturally detects redundancy (A and B triggering the same fault) and suppresses it, while promoting genuinely novel contributions. And the reserve ensures a safety floor of fault detection even in edge cases.

---

## 8. Connection to SETS and GD

### GD is a Special Case of DPP

SETS uses the Geometric Diversity metric:
```
GD(S) = log det(V_S × V_S^T)
```

This is a DPP with:
- Kernel `L = V × V^T` (VGG features only)
- No quality weighting (`q = [1, 1, 1, ...]`, all items equally weighted)
- Uncertainty handled externally via the separate fitness function

SETS then manually combines GD with uncertainty:
```
Fitness(x, S) = MaxP(x) × normalized(GD(S ∪ {x}) - GD(S))
```

### DPP Generalizes This

QDPS's L-ensemble `L = diag(q) × K × diag(q)` integrates quality into the kernel itself. The marginal gain `d_S[j]` in the L-ensemble already accounts for both the quality of item j and its diversity relative to S. There is no need for external combination, normalization, or the +0.5 bias term.

| Aspect | SETS (GD) | QDPS (DPP) |
|---|---|---|
| Diversity features | VGG only | VGG + output probabilities |
| Uncertainty handling | External (multiplied after normalization) | Internal (baked into kernel via quality scores) |
| Combination method | MaxP × normalized_GD_delta (heuristic) | det(L_S) (principled, single objective) |
| Selection strategy | Equidistant partitioning → local greedy | Global greedy on conditional variance |
| Optimality guarantee | None | (1 - 1/e) ≈ 63% |
| Marginal gain computation | slogdet per candidate, O(|S|³) each | Conditional variance update, O(1) each |

---

## 9. How QDPS Uses All of This

Putting it all together, QDPS works in four phases:

### Phase 1: Reserve
Reserve the top 10-20% most uncertain images. These are selected unconditionally to guarantee fault detection coverage.

### Phase 2: Reduce
Keep only the top α×k candidates by uncertainty. This shrinks the search space.

### Phase 3: Build Kernel
Construct the L-ensemble kernel:
```
K = λ × K_vgg + (1-λ) × K_prob       ← combined visual + behavioral similarity
q_i = uncertainty(image_i)^p           ← quality scores from uncertainty
L = diag(q) × K × diag(q) + εI        ← quality-weighted DPP kernel
```

Apply a penalty to candidates similar to reserved items (to avoid redundancy with the reserve).

### Phase 4: Select via Conditional Variance
```
Initialize d[j] = L[j,j]
For each slot in the remaining budget:
    Pick j* = argmax d[j]              ← best remaining candidate
    Update d[j] -= c[t,j]²            ← suppress similar candidates
Return reserved + selected
```

The conditional variance `d[j]` does all the heavy lifting:
- Items similar to already-selected ones have low d[j] → naturally avoided
- Items with low quality (uncertainty) have low d[j] → naturally avoided
- Items that are both high-quality and different from everything selected have high d[j] → naturally preferred

No normalization tricks, no separate weighting, no partitioning heuristics. One number per candidate, updated with one subtraction per step.

---

## 10. References

[1] A. Kulesza, B. Taskar. "Determinantal Point Processes for Machine Learning." Foundations and Trends in Machine Learning, Vol. 5, No. 2-3, pp. 123-286, 2012.
*The comprehensive reference for DPP theory, L-ensembles, quality-diversity decomposition, and MAP inference.*

[2] G. Nemhauser, L. Wolsey, M. Fisher. "An analysis of approximations for maximizing submodular set functions." Mathematical Programming, 14(1):265-294, 1978.
*The classic result proving the (1 - 1/e) guarantee for greedy maximization of submodular functions.*

[3] Y. Chen, Y. Zhang, A. Krause. "Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity." NeurIPS, 2018.
*Introduces the Cholesky-based incremental conditional variance updates used in QDPS.*

[4] J. Wang, H. Wu, P. Wang, X. Niu, C. Nie. "SETS: A Simple yet Effective DNN Test Selection Approach." ACM TOSEM, 2025.
*The baseline approach. Uses GD (a DPP determinant) without recognizing the DPP connection.*

[5] Z. Aghababaeyan, M. Abdellatif, M. Dadkhah, L. Briand. "DeepGD: A multi-objective black-box test selection approach for deep neural networks." ACM TOSEM, 2024.
*First to use GD for DNN test selection. Uses NSGA-II genetic algorithm to optimize uncertainty + GD jointly.*

[6] Z. Gong, P. Zhong, W. Hu. "Diversity in machine learning." IEEE Access, 7:64323-64350, 2019.
*Defines the Geometric Diversity (GD) metric and Standard Deviation (STD) diversity metric.*

[7] J. Shawe-Taylor, N. Cristianini. "Kernel Methods for Pattern Analysis." Cambridge University Press, 2004.
*Reference for kernel theory, including the closure of PSD kernels under convex combination.*

[8] W. Ma, M. Papadakis, A. Tsakmalis, M. Cordy, Y. Le Traon. "Test selection for deep learning systems." ACM TOSEM, 30(2):1-22, 2021.
*Defines the MaxP uncertainty metric and establishes its effectiveness for DNN test selection.*

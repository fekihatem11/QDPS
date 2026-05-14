# Why SETS Lacks Theoretical Guarantees — And Why It Matters

A detailed explanation of the optimality gap in SETS and how QDPS addresses it.

---

## 1. Setting the Scene: What Are We Optimizing?

When we select k test images from thousands of candidates, we want to maximize **fault diversity** — find as many different bugs in the model as possible. Since we can't know the actual faults without labels, we use a proxy called **Geometric Diversity (GD)**:

```
GD(S) = log det(V_S × V_S^T)
```

Think of it this way: each test image is a point in a high-dimensional space (its VGG16 feature vector). GD measures the **volume** of the shape formed by the selected points. If you select images that are all similar (clustered together), the volume is small. If you select images that are spread out in different directions, the volume is large.

Maximizing GD means selecting a spread-out, diverse set — which is what we want.

---

## 2. What is a Submodular Function?

GD (and log det in general) belongs to a special class of mathematical functions called **submodular functions**. Submodularity captures the idea of **diminishing returns**.

### An Everyday Analogy

Imagine you're building a team of experts to cover different skills:

- Hiring your first **programmer** is hugely valuable — you had no one who could code.
- Hiring a second **programmer** is still useful, but less so — you already have one.
- Hiring a third **programmer** adds even less — most coding needs are already covered.
- Meanwhile, hiring your first **designer** is hugely valuable again — it's a new skill entirely.

This is diminishing returns: the more you already have of something, the less value the next one adds.

### Formally

A function f is submodular if for any sets A ⊆ B and any item x not in B:

```
f(A ∪ {x}) - f(A)  ≥  f(B ∪ {x}) - f(B)
```

The marginal gain of adding x to a **smaller** set A is always at least as large as adding x to a **bigger** set B that contains A. In our context: adding a new diverse test image helps more when you have few images selected than when you already have many.

**GD (log det) has been proven to be submodular** [Nemhauser et al., 1978; Kulesza & Taskar, 2012]. This is not an assumption — it's a mathematical fact.

---

## 3. The Greedy Algorithm and Its Guarantee

### The Standard Greedy Algorithm

For submodular functions, there's a beautifully simple algorithm:

```
Start with S = {} (empty set)
Repeat k times:
    Look at ALL remaining candidates
    Pick the one that increases f(S) the most   ← "marginal gain"
    Add it to S
Return S
```

At each step, you make the single best possible addition considering everything that's already selected.

### The (1 - 1/e) Guarantee

In 1978, Nemhauser, Wolsey, and Fisher proved a landmark result:

> **Theorem**: For any monotone submodular function, the standard greedy algorithm produces a solution that is at least **(1 - 1/e) ≈ 63%** as good as the optimal solution.

What does this mean practically?

- Suppose the best possible selection of k images achieves GD = 100 (if an oracle told us the perfect answer)
- The greedy algorithm is guaranteed to achieve GD ≥ 63, no matter what
- This holds for **any** dataset, **any** model, **any** number of candidates

This is called an **approximation ratio** — a worst-case mathematical promise about solution quality. The algorithm might do better than 63% (and usually does), but it will **never** do worse.

### Why This Matters

Without a guarantee, an algorithm might work great on the datasets you tested but fail catastrophically on a new one. A guarantee gives you **confidence across all possible inputs**, which is essential for a method that claims to be general-purpose.

---

## 4. Why SETS Breaks the Guarantee

### What the Guarantee Requires

The (1 - 1/e) theorem has one critical requirement:

> At each step, you must evaluate **ALL remaining candidates** and pick the one with the largest marginal gain.

### What SETS Actually Does

SETS does not do this. Here's what happens:

**Step 1**: SETS takes the α×k candidates (say 300 for k=100) and splits them into k=100 chunks of 3 using equidistant interleaving:

```
Chunk  0: [candidate #0,  candidate #100, candidate #200]
Chunk  1: [candidate #1,  candidate #101, candidate #201]
Chunk  2: [candidate #2,  candidate #102, candidate #202]
...
Chunk 99: [candidate #99, candidate #199, candidate #299]
```

**Step 2**: For chunk 0, SETS evaluates only the 3 candidates in that chunk and picks the best.

**Step 3**: For chunk 1, SETS evaluates only the 3 candidates in that chunk and picks the best.

...and so on.

### Where the Guarantee Breaks

At step 2 (processing chunk 1), the **globally best candidate** might be candidate #202 sitting in chunk 2. But SETS never considers it — it only looks at the 3 candidates in chunk 1. It is forced to pick one of those 3, even if all three are mediocre.

The theorem says: "evaluate ALL remaining and pick the best." SETS says: "evaluate these 3 and pick the best of those." This is a fundamentally different algorithm, and the theorem does not apply to it.

### A Concrete Example of What Can Go Wrong

Imagine 9 candidates with these properties (k=3, α=3):

```
Candidate  Uncertainty  Fault type
─────────  ───────────  ──────────
x_0        0.95         Fault A
x_1        0.90         Fault A     ← same fault as x_0
x_2        0.85         Fault A     ← same fault as x_0
x_3        0.80         Fault B
x_4        0.75         Fault C
x_5        0.70         Fault D
x_6        0.65         Fault B     ← same fault as x_3
x_7        0.60         Fault C     ← same fault as x_4
x_8        0.55         Fault D     ← same fault as x_5
```

SETS partitions (interleaved):
```
Chunk 0: [x_0(A), x_3(B), x_6(B)]  → picks x_0 (Fault A)
Chunk 1: [x_1(A), x_4(C), x_7(C)]  → picks x_4 (Fault C)
Chunk 2: [x_2(A), x_5(D), x_8(D)]  → picks x_5 (Fault D)
```

Result: 3 faults detected (A, C, D). Fault B is missed.

**Global greedy** would have done:
```
Step 1: evaluate all 9, pick x_0 (Fault A, highest uncertainty + new fault)
Step 2: evaluate remaining 8, pick x_3 (Fault B, highest gain — A is already covered)
Step 3: evaluate remaining 7, pick x_4 (Fault C, highest gain — A and B covered)
```

Result: 3 faults detected (A, B, C). Also misses D, but the key point is that global greedy **adapts** — it avoids re-selecting Fault A candidates because their marginal diversity gain dropped after x_0 was selected. SETS can't adapt across chunks.

In this example both find 3 faults, but with different distributions of candidates the gap can widen. SETS has no theoretical bound preventing much worse outcomes.

---

## 5. How QDPS Restores the Guarantee

QDPS uses **greedy DPP MAP inference**, which is exactly the standard greedy algorithm applied to the log det objective:

```
Start with S = reserved inputs (high uncertainty guarantee)
Repeat until budget filled:
    Look at ALL remaining candidates
    Pick the one with the largest marginal gain in log det(L_S)
    Add it to S
```

This matches the theorem's requirement exactly. Therefore:

> QDPS inherits the (1 - 1/e) ≈ 63% approximation guarantee for its DPP selection phase.

### How It's Made Efficient

"Evaluate all remaining candidates at each step" sounds expensive — but the **Cholesky incremental update trick** [Chen et al., 2018] makes it fast:

Instead of recomputing det(L_S) from scratch every time (which would be O(k³) per step), QDPS maintains a running quantity called the **conditional variance** `d[j]` for each candidate j. This number tells us exactly how much adding j would increase log det — i.e., the marginal gain — without recomputing any determinant.

After selecting an item, updating all conditional variances takes O(N) time (one subtraction per remaining candidate). So:

- **Total cost**: O(N × k) for all k selection steps
- **Compared to SETS**: SETS computes slogdet (O(k²) each) for α candidates per chunk = O(α × k³) total
- **For large k, QDPS is faster AND has a guarantee**

---

## 6. Summary

| Property | SETS | QDPS |
|---|---|---|
| What it evaluates per step | 3 candidates (one chunk) | All remaining candidates |
| Selection strategy | Local greedy within fixed partition | Global greedy (standard algorithm) |
| Approximation guarantee | None (heuristic) | (1 - 1/e) ≈ 63% of optimal |
| Can miss the globally best candidate? | Yes (if it's in a different chunk) | No (all candidates considered) |
| Adapts to selections so far? | Only within current chunk | Yes, fully (conditional variances update globally) |
| Complexity | O(α × k³) for slogdet per chunk | O(N × k) with Cholesky updates |

The key takeaway: SETS' equidistant partitioning is a speed optimization that sacrifices the theoretical guarantee. QDPS achieves the same speed (actually faster for large k) while keeping the guarantee, by using Cholesky incremental updates instead of partitioning to reduce computation.

---

## References

- Nemhauser, Wolsey, Fisher. "An analysis of approximations for maximizing submodular set functions." *Mathematical Programming*, 1978.
- Kulesza, Taskar. "Determinantal Point Processes for Machine Learning." *Foundations and Trends in ML*, 2012.
- Chen, Zhang, Krause. "Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity." *NeurIPS*, 2018.

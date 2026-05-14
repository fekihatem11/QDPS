# The Connection Between GD and DPP

## Two Formulas, One Object

Geometric Diversity (GD) and Determinantal Point Processes (DPP) were developed in completely different fields for completely different purposes. But they share the same core mathematical object.

```
GD(S)  = det(V_S × V_S^T)       ← from diversity measurement literature
P(S)  ∝ det(L_S)                 ← from probability theory (DPP)
```

Both compute the **determinant of a matrix built from item features**. The matrix `V_S × V_S^T` is a Gram matrix — and a Gram matrix is exactly the kind of kernel matrix L that defines a DPP.

In other words: **GD is a DPP. Maximizing GD is equivalent to finding the MAP (most probable) subset under a DPP.**

---

## A Worked Example

### Setup: 3 Test Images

Suppose we have 3 test images, each represented by a 2D feature vector (for simplicity):

```
Image A:  v_A = [1.0,  0.0]    ← points right
Image B:  v_B = [0.0,  1.0]    ← points up
Image C:  v_C = [0.9,  0.1]    ← points almost right (similar to A)
```

Visually:

```
        B [0,1]
        ↑
        |
        |     C [0.9, 0.1]
        |   /
        | /
        +--------→ A [1,0]
```

A and C point in nearly the same direction — they are similar. B points in a completely different direction — it is diverse from both A and C.

---

### Step 1: Computing GD for Each Pair

To compute GD, we build `V_S × V_S^T` (the Gram matrix) for each pair and take its determinant.

#### Pair {A, B} — diverse pair

```
V = [ 1.0  0.0 ]       V × V^T = [ 1.0×1.0 + 0.0×0.0    1.0×0.0 + 0.0×1.0 ]
    [ 0.0  1.0 ]                  [ 0.0×1.0 + 1.0×0.0    0.0×0.0 + 1.0×1.0 ]

                                = [ 1.0   0.0 ]
                                  [ 0.0   1.0 ]

GD({A,B}) = det = 1.0 × 1.0 - 0.0 × 0.0 = 1.0
```

The Gram matrix is the identity — A and B are perfectly orthogonal (zero similarity). The determinant is **maximal**.

#### Pair {A, C} — similar pair

```
V = [ 1.0  0.0 ]       V × V^T = [ 1.0×1.0 + 0.0×0.0    1.0×0.9 + 0.0×0.1 ]
    [ 0.9  0.1 ]                  [ 0.9×1.0 + 0.1×0.0    0.9×0.9 + 0.1×0.1 ]

                                = [ 1.00   0.90 ]
                                  [ 0.90   0.82 ]

GD({A,C}) = det = 1.00 × 0.82 - 0.90 × 0.90 = 0.82 - 0.81 = 0.01
```

The off-diagonal (0.90) is large — A and C are very similar. The determinant is **tiny**.

#### Pair {B, C} — moderately diverse pair

```
V = [ 0.0  1.0 ]       V × V^T = [ 0.0×0.0 + 1.0×1.0    0.0×0.9 + 1.0×0.1 ]
    [ 0.9  0.1 ]                  [ 0.9×0.0 + 0.1×1.0    0.9×0.9 + 0.1×0.1 ]

                                = [ 1.00   0.10 ]
                                  [ 0.10   0.82 ]

GD({B,C}) = det = 1.00 × 0.82 - 0.10 × 0.10 = 0.82 - 0.01 = 0.81
```

The off-diagonal (0.10) is small — B and C are fairly different. The determinant is **large**.

#### Summary

```
Pair      GD (determinant)   Interpretation
────      ────────────────   ──────────────
{A, B}         1.00          Maximum diversity (orthogonal vectors)
{B, C}         0.81          High diversity (B is different from C)
{A, C}         0.01          Almost no diversity (A ≈ C)
```

---

### Step 2: Now Read This as a DPP

The DPP says: `P(S) ∝ det(L_S)`.

If we use the Gram matrix as our kernel L (i.e., `L = V × V^T`), then the DPP probability of each pair is proportional to the GD value we just computed:

```
P({A,B}) ∝ 1.00
P({B,C}) ∝ 0.81
P({A,C}) ∝ 0.01
```

Normalizing (dividing by the sum 1.00 + 0.81 + 0.01 = 1.82):

```
P({A,B}) = 1.00 / 1.82 = 54.9%
P({B,C}) = 0.81 / 1.82 = 44.5%
P({A,C}) = 0.01 / 1.82 =  0.5%
```

The DPP overwhelmingly prefers {A,B} and {B,C} — the diverse pairs. It almost never selects {A,C} — the redundant pair. **This is exactly what GD ranks as well.**

Maximizing GD picks {A,B}. The DPP MAP (most probable subset) is also {A,B}. They are the same optimization.

---

### Step 3: The Geometric Meaning

The determinant of the Gram matrix equals the **squared volume** of the parallelogram (or parallelepiped in higher dimensions) formed by the vectors.

```
Pair {A, B}:                    Pair {A, C}:

    B ↑                              C  /
      |                              | / (almost flat)
      |                              |/
      +———→ A                        +———→ A

 Area = 1.0 (full square)       Area = 0.1 (thin sliver)
 det  = 1.0                     det  = 0.01 (area squared)
```

- **Diverse vectors** span a large area → large determinant → high GD → high DPP probability
- **Similar vectors** form a flat shape → tiny area → small determinant → low GD → low DPP probability

---

## Why This Connection Matters

Once you recognize that GD = DPP, three things unlock:

| What GD alone gives you | What the DPP framework adds |
|---|---|
| A diversity score | A **probability distribution** over subsets |
| No guidance on how to optimize | **Submodularity** → greedy gives (1-1/e) guarantee |
| Diversity only | **L-ensemble** → bake quality (uncertainty) into the kernel |
| Recompute full determinant each step | **Cholesky updates** → O(N×k) efficient greedy |

GD tells you *what* to measure. DPP tells you *how to optimize it efficiently, with guarantees, while also incorporating quality*.

---

## From GD to QDPS: The Progression

```
Step 1 — GD:          det(V × V^T)              Pure diversity, no quality
                            ↓
Step 2 — Recognize:   GD = DPP with L = V×V^T    Same math, richer framework
                            ↓
Step 3 — L-ensemble:  L = diag(q) × K × diag(q)  Quality baked into the kernel
                            ↓
Step 4 — Greedy MAP:  Pick argmax log(d[j])       Efficient O(N×k) selection
                            ↓
Step 5 — QDPS:        Combined kernel + reserve    Full method
```

The entire progression starts from one observation: GD and DPP compute the same determinant.

# MAP Inference and DPP: How We Find the Best Subset

---

## 1. What is Inference?

In everyday language, "inference" means drawing a conclusion from available information. In math and statistics, it means the same thing but more precisely: **using a model to answer a question about data.**

A model is something that encodes knowledge. A question is what you want to know. Inference is the process of getting the answer.

Examples:
- **Model**: a weather forecast system. **Question**: "will it rain tomorrow?" **Inference**: computing the probability of rain.
- **Model**: a trained neural network. **Question**: "is this image a cat or a dog?" **Inference**: running the image through the network and reading the output.
- **Model**: a DPP (defined by matrix L). **Question**: "which subset of k items is the best?" **Inference**: finding the subset with the highest determinant.

### Types of Inference

Given a probability model, there are two main things you can ask:

**1. Sampling**: "Give me a random outcome, respecting the probabilities."

You don't want a specific answer — you want variety. Each time you sample, you might get a different result.

Example: a music app that generates a different playlist each time you open it. You want diversity, but not always the same list.

**2. Optimization (MAP)**: "Give me the single best outcome — the one with the highest probability."

You want one definitive answer — the best one.

Example: you have a budget to test 100 images. You want **the** best 100, not a random diverse set. You'll run this once and use the result.

---

## 2. What Does MAP Stand For?

**MAP = Maximum A Posteriori**

Breaking it down word by word:

- **Maximum**: we want to find the thing that makes something as large as possible
- **A Posteriori**: a Latin phrase meaning "from what comes after" — it means "given the model we have built"

So MAP inference means: **given our model, find the outcome that the model considers most likely.**

### MAP in Different Contexts

MAP is a general concept used across many areas:

- In image reconstruction: "given a noisy image, find the most likely clean image"
- In speech recognition: "given an audio signal, find the most likely sentence"
- In DPP: "given the kernel matrix L, find the subset with the highest probability"

The word MAP doesn't change meaning — it always means "find the most probable outcome." What changes is the model and what the outcomes look like.

---

## 3. MAP Inference in a DPP

### What the DPP Model Tells Us

A DPP with kernel L assigns a probability to every possible subset:

```
P(X = S) = det(L_S) / det(L + I)
```

Subsets with large determinants get high probability. Subsets with small determinants get low probability. And we know that determinants are large when items are high-quality and diverse.

### What MAP Asks

MAP asks: **which subset S of size k has the highest probability?**

```
S* = argmax  P(X = S)
     |S|=k

   = argmax  det(L_S) / det(L + I)
     |S|=k
```

Since det(L + I) is the same constant for every subset, it doesn't affect which subset wins. So we can drop it:

```
S* = argmax  det(L_S)
     |S|=k
```

This is equivalent to:

```
S* = argmax  log det(L_S)
     |S|=k
```

because log is monotonically increasing — if det(L_A) > det(L_B), then log det(L_A) > log det(L_B). The ranking doesn't change.

We prefer log det because:
- Determinants can be astronomically large or tiny numbers — log keeps them manageable
- log det is the function that has the submodularity property (needed for the greedy guarantee)

### What MAP Gives Us

The MAP subset is the **single most probable** subset under the DPP. Because the DPP assigns high probability to diverse, high-quality subsets, the MAP subset is the most diverse, high-quality subset of size k.

This is exactly what we want for test case prioritization: one definitive set of k test cases that maximizes both fault-revealing potential and coverage of different fault types.

---

## 4. Why MAP Inference is Hard

### The Brute Force Approach

The obvious approach: try every possible subset of size k, compute its determinant, pick the largest.

How many subsets are there? The number of ways to choose k items from N is:

```
C(N, k) = N! / (k! × (N-k)!)
```

Some real numbers:

```
N       k       C(N,k)
───     ──      ──────
10      3       120                    ← easy
100     10      17,310,309,456,440     ← 17 trillion
300     100     ~10^81                 ← more than atoms in the universe
1000    100     ~10^139                ← absurdly impossible
```

For any realistic problem, trying all subsets is impossible.

### MAP is NP-Hard

Computer scientists have proven that finding the exact MAP subset for a DPP is **NP-hard**. This means there is no known algorithm that can guarantee finding the absolute best subset in polynomial time for all inputs.

But this doesn't mean we're stuck. It means we need to accept an **approximate** solution — one that is provably close to optimal, even if not perfect.

---

## 5. The Greedy Approach

### The Idea

Instead of searching over all possible subsets, build the subset **one item at a time**. At each step, add the item that improves the objective the most.

```
Start with S = {} (empty set)

Step 1: Try adding each of the N items individually.
        Pick the one that gives the largest det(L_S).
        → S = {best single item}

Step 2: Try adding each of the remaining N-1 items.
        Pick the one that increases det(L_S) the most.
        → S = {best single item, best second item}

Step 3: Try adding each of the remaining N-2 items.
        Pick the one that increases det(L_S) the most.
        → S = {item, item, item}

...repeat until |S| = k
```

### Why Greedy Works Here: Submodularity

Greedy doesn't always give good results. For some optimization problems, being greedy at each step leads to a terrible overall solution. But for DPP MAP inference, greedy is provably good. The reason is **submodularity**.

#### What is Submodularity?

A function f(S) is submodular if adding an item helps **less** when the set is already large. This is called the **diminishing returns** property.

Formally: for any sets A ⊆ B (A is a subset of B) and any item j not in B:

```
f(A ∪ {j}) - f(A)  ≥  f(B ∪ {j}) - f(B)
    ↑                       ↑
  gain of adding j        gain of adding j
  to the SMALLER set      to the LARGER set
```

Adding j to the smaller set A always helps at least as much as adding j to the larger set B.

#### Everyday Example

You're furnishing an empty apartment:
- Buying the **first chair** is a huge improvement — you go from standing to sitting
- Buying the **second chair** is nice — a guest can sit too
- Buying the **tenth chair** adds almost nothing — you already have plenty

Each additional chair helps less. That's diminishing returns.

#### Why log det is Submodular

The function f(S) = log det(L_S) is submodular because of conditional variance.

Adding item j to set S increases log det by:

```
f(S ∪ {j}) - f(S) = log(d_S[j])
```

where d_S[j] is the conditional variance — how much new information j brings given what's already in S. As S grows, more of j's information is already covered, so d_S[j] can only **decrease or stay the same**. The marginal gain diminishes.

Concrete example:
- Adding a "cat photo" test to an empty set → huge gain (brand new information)
- Adding a "cat photo" test when you already have 5 cat photo tests → tiny gain (most cat-related information is already covered)

#### The Guarantee

Nemhauser, Wolsey, and Fisher (1978) proved that for any non-negative monotone submodular function, greedy achieves:

```
f(S_greedy) ≥ (1 - 1/e) × f(S_optimal)  ≈  0.632 × f(S_optimal)
```

This means the greedy solution's log det is at least 63.2% of the optimal log det. **No matter how hard the problem, greedy gets you at least this far.**

In practice, greedy usually does **much better** than 63% — this is a worst-case floor, not a typical result.

---

## 6. The Naive Greedy Algorithm

### Step by Step

```
Input: L matrix (N×N), budget k
Output: selected set S of size k

S = {}
For t = 1 to k:
    For each remaining candidate j:
        Compute marginal_gain(j) = log det(L_{S∪{j}}) - log det(L_S)
    Pick j* = the candidate with the highest marginal_gain
    S = S ∪ {j*}
Return S
```

### The Problem: Speed

At each step t, for each remaining candidate, we need to compute det(L_{S∪{j}}). Computing a determinant of a (t+1) × (t+1) matrix costs O(t³). We do this for ~N candidates at each of k steps:

```
Total cost = N × (1³ + 2³ + 3³ + ... + k³) ≈ N × k⁴ / 4
```

For N = 1000 and k = 100: approximately 2.5 × 10^11 operations. Too slow for practical use.

---

## 7. The Efficient Greedy Algorithm: Conditional Variance

### The Key Mathematical Identity

There exists an identity that converts the expensive determinant comparison into a simple number lookup:

```
log det(L_{S∪{j}}) - log det(L_S) = log(d_S[j])
```

where d_S[j] is the **conditional variance** of item j given the selected set S.

This means: **finding the item with the largest marginal gain is the same as finding the item with the largest conditional variance.** We never need to compute determinants.

### What is Conditional Variance?

d_S[j] answers the question: **"if I already selected the items in S, how much genuinely new information does item j bring?"**

Start with a simple idea: every item carries some **total information** — represented by L[j,j] (its diagonal entry in the L matrix). Think of this as a bucket of water. A full bucket means lots of information.

#### Before Any Selection

Nothing is selected yet. Every item has its full bucket:

```
d[A] = L[A,A] = 0.36    ← full bucket
d[B] = L[B,B] = 0.30    ← full bucket
d[C] = L[C,C] = 0.25    ← full bucket
```

At this point, d[j] = L[j,j] for every item. All information is "new" because there's nothing to be redundant with.

#### After Selecting A

Now A is in the set. Some of A's information **overlaps** with B's and C's. That overlapping part is no longer "new" — it's already covered by A.

The conditional variance is: **what's left in the bucket after you pour out the part that's already covered.**

```
d[B] = 0.30 - 0.25 = 0.05
       ────   ────
       total  overlap with A
       info   (B and A are very similar,
              so most of B's info is redundant)

d[C] = 0.25 - 0.04 = 0.21
       ────   ────
       total  overlap with A
       info   (C and A are different,
              so very little is redundant)
```

B lost most of its water — it had almost the same information as A. C kept most of its water — it carries genuinely different information.

#### Why It's Called "Conditional"

The word "conditional" means **"given that something else is already known."**

- d[B] **before** any selection = 0.30 → B's total information
- d[B] **conditional on A being selected** = 0.05 → B's information **given that we already have A**

It's the same concept as everyday conditionals:
- "How useful is an umbrella?" → Very useful (it protects from rain)
- "How useful is an umbrella, **given that you already have one**?" → Almost useless (redundant)

The umbrella didn't change. Its **conditional** usefulness changed because of what you already have.

#### Why It's Called "Variance"

This comes from statistics. Variance measures **how much unpredictability remains**. If you can perfectly predict something, its variance is zero — no surprise left.

- If B is very similar to A, then once you know A, you can almost perfectly predict B → low remaining unpredictability → low conditional variance
- If C is very different from A, then knowing A tells you almost nothing about C → high remaining unpredictability → high conditional variance

High conditional variance = **this item would surprise you** = it carries information you don't already have = **it's worth selecting**.

#### The Formula

```
d_S[j] = L[j,j]  -  L[j,S] × L[S,S]^(-1) × L[S,j]
         ──────     ─────────────────────────────────
         total      the part of j that S already
         bucket     explains (the overlap)
```

The second term is a projection — it computes exactly how much of j's information lives in the "space" spanned by the already-selected items. Subtract that, and what remains is the genuinely new part.

#### Summary

- **j is very different from everything in S**: the overlap is small → d_S[j] is large → j is a good candidate
- **j is very similar to something in S**: the overlap is almost equal to L[j,j] → d_S[j] is near zero → j is a bad candidate (it would be wasted budget)

At each step, the greedy algorithm picks the item with the **largest conditional variance** — the item with the most water left in its bucket — the item that brings the most new information. This is equivalent to picking the item that increases det(L_S) the most.

### The Link to Determinants

Why does log(d_S[j]) equal the marginal gain in log det?

Think of the determinant as a **volume**. When you add item j to set S, the volume increases by the "height" of j above the space spanned by S. That height is exactly the conditional variance — it measures how much j sticks out in a direction not covered by S.

- If j lies entirely within the space spanned by S (similar to existing items), its height is zero → no volume increase → zero marginal gain
- If j points in a completely new direction, its height is maximal → large volume increase → large marginal gain

### The Cholesky Incremental Update

Computing d_S[j] from the formula above still requires a matrix inverse. The Cholesky trick (Chen et al., 2018) avoids this by maintaining d[j] **incrementally** — updating it with a single subtraction after each selection.

**Initialization** (before any selection):

```
d[j] = L[j,j]    for all j = 1, ..., N
```

When S is empty, every item's conditional variance equals its diagonal entry — its total information. Nothing is redundant yet.

**At step t, after selecting item j***, update two things:

1. Compute the Cholesky factor c[t,:]:

```
If t = 1 (first selection):
    c[1, j] = L[j*, j] / sqrt(d[j*])

If t > 1:
    c[t, j] = (L[j*, j] - sum_{s=1}^{t-1} c[s,j*] × c[s,j]) / sqrt(d[j*])
```

The Cholesky factor c[t,j] measures **how much item j overlaps with the newly selected item j***.

2. Update conditional variances:

```
d[j] = d[j] - c[t,j]²    for all remaining j
```

One subtraction per candidate. That's it.

### What Happens When You Update d[j]

The subtraction `d[j] = d[j] - c[t,j]²` removes the part of j's information that overlaps with the newly selected item j*.

**If j is similar to j***: c[t,j] is large → c[t,j]² is large → d[j] drops significantly. The DPP has "learned" that j is redundant.

**If j is different from j***: c[t,j] is small → c[t,j]² is tiny → d[j] barely changes. Item j remains a strong candidate.

### Worked Example

5 candidates, budget k = 2. Diagonal of L:

```
Initial:  d[A] = 0.36    d[B] = 0.30    d[C] = 0.25    d[D] = 0.20    d[E] = 0.01
```

**Step 1**: Pick A (highest d = 0.36)

Compute c[1,:] and update d:

```
Suppose A is very similar to B, slightly similar to C, and different from D and E:

c[1,B] = 0.50    →    d[B] = 0.30 - 0.50² = 0.30 - 0.25 = 0.05   ← collapsed (redundant with A)
c[1,C] = 0.20    →    d[C] = 0.25 - 0.20² = 0.25 - 0.04 = 0.21   ← barely changed
c[1,D] = 0.10    →    d[D] = 0.20 - 0.10² = 0.20 - 0.01 = 0.19   ← barely changed
c[1,E] = 0.05    →    d[E] = 0.01 - 0.05² = 0.01 - 0.0025 = 0.0075
```

**After step 1**: d[C] = 0.21, d[D] = 0.19, d[B] = 0.05, d[E] = 0.0075

**Step 2**: Pick C (highest d = 0.21)

Result: S = {A, C}

Notice: B had higher initial quality than C and D (0.30 > 0.25 > 0.20), but after A was selected, B's conditional variance collapsed to 0.05 because B and A carry the same information. C and D, being different from A, kept most of their value.

**This is the entire point of conditional variance**: it automatically detects and penalizes redundancy without any manual rules.

---

## 8. The Complete Algorithm

```
Input:  L matrix (N×N), budget k
Output: selected set S

1. Initialize:
   d[j] = L[j,j]  for all j        ← total information of each item
   c = empty matrix (k × N)         ← will store Cholesky factors
   S = {}
   remaining = {1, 2, ..., N}

2. For t = 1 to k:

   a. SELECT: Pick j* = argmax d[j] over remaining items
      (scan all remaining d values, pick the largest)

   b. Add j* to S, remove from remaining

   c. UPDATE Cholesky factor:
      If t = 1:
          c[1, j] = L[j*, j] / sqrt(d[j*])    for all remaining j
      Else:
          c[t, j] = (L[j*, j] - sum_{s=1}^{t-1} c[s,j*] × c[s,j]) / sqrt(d[j*])

   d. UPDATE conditional variances:
      d[j] = d[j] - c[t,j]²    for all remaining j

3. Return S
```

### Complexity

```
Step 2a: O(N)     ← scan remaining items
Step 2c: O(N × t) ← compute c[t,j] for each remaining item (inner product of length t)
Step 2d: O(N)     ← one subtraction per item

Total across all k steps: O(N × k²)
```

With the fast variant (precomputing certain products): O(N × k).

Compare:
```
Brute force (try all subsets):     C(N,k) — impossibly large
Naive greedy (recompute det):      O(N × k⁴)
Cholesky greedy:                   O(N × k²) or O(N × k)
```

For N = 1000, k = 100:
```
Brute force:    ~10^139 operations    (impossible)
Naive greedy:   ~10^11 operations     (very slow)
Cholesky greedy: ~10^5 operations     (instant)
```

---

## 9. The Full Picture: From DPP to Selected Test Cases

Here is how everything connects, from start to finish:

```
YOU HAVE                          YOU BUILD                        YOU GET
──────────                        ─────────                        ───────

Feature vectors    ──┐
(VGG, output probs)  ├──→  Kernel K (similarity)  ──┐
                     │                                │
Uncertainty scores ──┘     Quality q (from MaxP)  ───┤
                                                     │
                                                     ▼
                                              L = diag(q) × K × diag(q)
                                              (L-ensemble kernel)
                                                     │
                                                     ▼
                                              Initialize d[j] = L[j,j]
                                                     │
                                                     ▼
                                              Greedy loop:
                                                pick max d[j]
                                                update d[j] -= c[t,j]²
                                                     │
                                                     ▼
                                              S* = {selected test cases}
                                              (high quality + diverse)
```

### What Each Piece Does

| Component | Role | What would happen without it |
|---|---|---|
| **L matrix** | Encodes quality + diversity in one object | You'd need to combine two separate scores ad-hoc (like SETS does) |
| **det(L_S)** | Jointly measures quality and diversity of a subset | You'd need a manual formula to balance the two |
| **MAP inference** | Finds the subset maximizing det(L_S) | You'd have no principled way to select the best subset |
| **Submodularity** | Guarantees greedy is near-optimal | You'd have no confidence that greedy gives a good answer |
| **Conditional variance d[j]** | Measures new information of each candidate | You'd need to recompute full determinants (too slow) |
| **Cholesky update** | Makes d[j] updates O(1) per item | d[j] computation would require matrix inversion (too slow) |

### Why Not Just Sort by Uncertainty?

If you sort by MaxP and take the top k, you get the k most uncertain images. But many of them might trigger the **same fault**. For example, 50 blurry photos of the digit "3" that the model confuses with "8" — all highly uncertain, but all redundant.

MAP inference over a DPP avoids this. After selecting the first blurry "3", the conditional variance of all similar blurry "3"s collapses. The algorithm moves on to images that trigger **different** faults.

### Why Not Just Maximize Diversity?

If you only maximize diversity (ignoring quality), you might select images the model classifies **correctly** — just because they're visually different. That wastes budget on test cases that reveal no faults.

The L-ensemble solves this: quality enters through the diagonal (L[j,j] = q_j²), so low-quality items have tiny conditional variance from the start and never get selected.

### The Balance

MAP inference on the L-ensemble DPP finds the subset where:
- Every item is individually high-quality (likely to reveal a fault)
- Every pair of items is diverse (likely to reveal **different** faults)
- The balance between these two objectives emerges from the determinant — no manual tuning

---

## 10. Summary

| Question | Answer |
|---|---|
| What is inference? | Using a model to answer a question |
| What is MAP? | Finding the single most probable outcome under a model |
| What is the DPP model? | P(S) ∝ det(L_S) — diverse subsets get higher probability |
| What does MAP give us? | The subset of size k with the largest determinant = most diverse + highest quality |
| Why not try all subsets? | C(N,k) is astronomically large — impossible |
| Why does greedy work? | log det is submodular (diminishing returns) → greedy ≥ 63.2% of optimal |
| How is greedy made fast? | Conditional variance d[j] replaces determinant computation. Cholesky updates make it O(N×k) |
| What does d[j] measure? | How much new information item j would bring, given what's already selected |
| How is d[j] updated? | d[j] = d[j] - c[t,j]² — one subtraction per item per step |

---

## References

[1] A. Kulesza, B. Taskar. "Determinantal Point Processes for Machine Learning." Foundations and Trends in Machine Learning, Vol. 5, No. 2-3, pp. 123-286, 2012.

[2] G. Nemhauser, L. Wolsey, M. Fisher. "An analysis of approximations for maximizing submodular set functions." Mathematical Programming, 14(1):265-294, 1978.

[3] Y. Chen, Y. Zhang, A. Krause. "Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity." NeurIPS, 2018.

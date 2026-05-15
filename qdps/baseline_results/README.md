# `qdps/baseline_results/`

Per-method, per-subject experimental output from the SETS replication
package (Wang et al., TOSEM 2025) — RQ2 & RQ3 measurements. Used by
`qdps/statistical_test.py` to run paired Wilcoxon signed-rank tests of
QDPS *vs* each baseline.

---

## Layout

```
qdps/baseline_results/
├── SETS/
│   └── <subject>/
│       └── selected_indices_k{100,300,500}_run{0..29}.npy
├── DeepGD/
│   └── <subject>/
│       └── selected_indices_k{100,300,500}_run{0..29}.npy
└── RS/    (Random Selection)
    └── <subject>/
        └── selected_indices_k{100,300,500}_run{0..29}.npy
```

Each `.npy` file is the **list of test-input indices selected** by that
method on that subject at that budget for that run. Length = budget (k).
30 runs per (method, subject, budget) combination.

---

## Provenance

Copied verbatim from `SETS/Experiment_results/RQ2&3/` in the official
SETS replication package.

The selections were produced against the original SETS pretrained models
(see `qdps/fault_clusters/` for the matching ground-truth fault clusters).
Reproducing this requires the same pretrained model files, which are NOT
inside this folder — they live in the SETS-paper Google Drive distribution.
Our robustness pipeline retrains its own model instances from scratch
(`qdps/robustness/<subject>/instances/seed_<n>/model.h5`), independent of
this data.

---

## How it's used

```python
# qdps/statistical_test.py loads each baseline's selected indices,
# computes FDR per run using qdps/fault_clusters/<subject>/, and
# compares against QDPS's per-run FDR distribution with a paired
# Wilcoxon signed-rank test.
```

Headline statistics from this data are duplicated in `qdps/BASELINES.md`
(SETS / DeepGD / RS published FDRs by subject and budget). If anything
here changes, update `BASELINES.md` too.

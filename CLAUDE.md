# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

DNN test case prioritization (TCP) research for image classification. The goal is to publish a new **black-box** test selection method (**QDPS** — Quality-Diversity DPP Selection) that beats **SETS** (Wang et al., TOSEM 2025) on Fault Detection Rate (FDR) across 8 dataset/model subjects × 3 budgets (k=100, 300, 500) = 24 configurations.

Background lives in `docs/RESEARCH_LOG.md` (experiment history) and `docs/METHOD_EXPLANATION.md` / `docs/DPP_EXPLAINED.md` / `docs/MAP_INFERENCE_EXPLAINED.md` (theory). The SETS paper's replication package is vendored under `SETS/`.

## Common commands

The canonical entry point is the Makefile in `qdps/`. Run from inside `qdps/`:

```
make compare       # Run QDPS once and print comparison vs SETS / DeepGD / RS (paper FDRs)
make run           # Full experiment: QDPS x 8 subjects x 3 budgets x 30 runs
make quick         # 5 runs per configuration
make single        # 1 run per configuration
make sets          # Run SETS baseline (30 runs) -> qdps/sets_results/
make stat          # Wilcoxon signed-rank test: QDPS vs SETS / DeepGD / RS
make subject S=mnist_LeNet1 N=30 K=100,300,500    # Single subject
make clean         # rm -rf results/
```

Available subjects (use these exact strings for `S=`):
`mnist_LeNet1`, `mnist_LeNet5`, `Fashion_mnist_LeNet4`, `cifar10_12Conv`, `cifar10_ResNet20`, `SVHN_LeNet5`, `Fruit360_ResNet50`, `TinyImageNet_ResNet101`.

Generic method runner (in `scripts/`, picks up any module from `methods/`):

```
python scripts/run_experiment.py <method_module> [n_runs]   # e.g. dpp_greedy, qdps, sets_baseline
python scripts/compare_results.py                           # QDPS vs all paper baselines
```

Dependencies live in `SETS/requirements.txt` (conda export with TF/Keras pinned for the original SETS code — only `numpy`/`scipy`/`scikit-learn`/`pickle` are actually needed for QDPS).

## Architecture

There are **two parallel scaffoldings** for the same research workflow. Don't mix them up:

- **`qdps/`** — current consolidated home for the proposed method. Self-contained: `qdps.py` (algorithm), `data_loader.py`, `run_experiment.py`, `run_single_subject.py`, `run_sets_baseline.py`, `statistical_test.py`, `generate_*_table.py`, plus a `Makefile`. Results go to `qdps/results/` and `qdps/sets_results/`.
- **`methods/` + `scripts/`** — older method-comparison framework. `methods/` holds candidate algorithms tried during exploration (`dpp_greedy.py`, `hybrid_dpp.py`, `sets_enhanced.py`, `adaptive_pads.py`, `facility_location.py`, `cluster_stratified.py`, `prob_diversity.py`, …). `scripts/run_experiment.py` dynamically imports any of them by module name. Results go to `experiments/results/<METHOD>_<TIMESTAMP>/`.

`methods/qdps.py` and `qdps/qdps.py` are the same algorithm — keep them in sync when editing.

### Method module contract

Every algorithm under `methods/` exposes:

```python
METHOD_NAME: str
def select(size, index, features, output_probability) -> list[int]
```

Return value is the list of selected test-input indices. The runner times the call and computes FDR.

### Data flow

Input data is pre-processed by the SETS authors and lives in `SETS/Input_data/Fault_clusters/<subject>/`:
- `output_probability.npy` — model softmax outputs (n_samples × n_classes)
- `cluster_results.npy` (`.pkl` for TinyImageNet) — DBSCAN cluster label per misclassified input; `-1` = noise
- `mis_index_test.npy` (`.pkl` for TinyImageNet) — indices of misclassified inputs
- VGG16 features live one level up in `SETS/features/features_test_<subject>.npy` — `data_loader.load_subject` falls back to that location when `features_test.npy` is missing in the subject folder

`Fault_clusters` folder names don't all match `data_name_model_name` exactly (`cifar10_12conv` is lowercase, `Fruit360_ResNet50` etc.). Use `FOLDER_MAP` in `data_loader.py` — don't construct paths by hand.

FDR computation (`compute_fdr`) follows the SETS paper's **"1noisy"** convention: all noise-cluster (-1) misclassifications collapse to a single "fault", and `FDR = unique_faults_found / min(budget, total_faults)`. Don't change this — it's what the published numbers are measured against.

### QDPS algorithm (the load-bearing idea)

Quality-weighted DPP MAP inference with a **mixed similarity kernel**:

1. **Uncertainty (MaxP)**: `q_i = (1 - max(p_i))^quality_power`
2. **Reserve**: top `reserve_ratio*k` highest-uncertainty inputs are guaranteed selection (covers small-budget failure mode of pure DPP)
3. **Candidate pool**: top `alpha*k` by uncertainty (minus reserved)
4. **Kernel**: `K = feature_mix * K_vgg + (1 - feature_mix) * K_prob`, where `K_prob` is cosine similarity over **log-probability vectors** (this is the key novelty — behavioral diversity targets fault diversity directly).
5. **L-ensemble**: `L = diag(q) K diag(q)`, with a similarity penalty against the reserved set.
6. **Greedy MAP**: incremental Cholesky updates, O(Nk²).

Hyperparameters are **budget-adaptive** (see `ADAPTIVE_DEFAULTS` in `qdps.py`). Don't hardcode a single setting — the small/medium/large regimes were tuned per budget.

## Conventions

- Published baseline FDRs (SETS / DeepGD / RS paper Table 3 numbers) are duplicated in `qdps/BASELINES.md`, `qdps/run_single_subject.py`, `qdps/compare_results.py`, and `scripts/compare_results.py`. If any of them changes, update all four.
- Win/loss thresholds use a **±0.005 FDR band** for "tie" (see `compare_with_baseline` / `run_qdps_comparison`).
- Experiment outputs always include `meta` (method, timestamp, n_runs, budgets, subjects) plus per-run FDR/time arrays so results stay reproducible.
- This directory is **not a git repository**; `SETS/` is a vendored sub-repo with its own `.git/`. Don't run `git init` at the top level without asking.

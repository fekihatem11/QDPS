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

All input data needed by the main QDPS comparison lives **inside `qdps/`** (the original SETS replication package under `SETS/` is no longer referenced by any code path). The relevant folders:

- `qdps/fault_clusters/<subject>/` — ground-truth fault structure for each subject (was `SETS/Input_data/Fault_clusters/`). Files:
  - `output_probability.npy` — model softmax outputs (n_samples × n_classes)
  - `cluster_results.npy` (`.pkl` for TinyImageNet) — cluster label per misclassified input; `-1` = noise
  - `mis_index_test.npy` (`.pkl` for TinyImageNet) — indices of misclassified test inputs
- `qdps/features_for_selection/features_test_<subject>.npy` — VGG16 features (min-max normalized, X_scf variant) used by the QDPS / SETS *selection* kernel. `data_loader.load_subject` falls back here when `features_test.npy` is missing in a subject folder.
- `qdps/baseline_results/{SETS,DeepGD,RS}/<subject>/` — published per-method per-run selection outputs (was `SETS/Experiment_results/RQ2&3/`). Used by `statistical_test.py`.
- `qdps/robustness/features_for_clustering/<dataset>/features_{train,test}_raw.npy` — raw VGG16 features used by the **fault-clustering** pipeline (UMAP + HDBSCAN), one subfolder per dataset (`mnist/`, `cifar10/`, …), separate from the selection-kernel features above because clustering needs un-normalized output.

`fault_clusters/` folder names don't all match `data_name_model_name` exactly (`cifar10_12conv` is lowercase, `Fruit360_ResNet50` etc.). Use `FOLDER_MAP` in `data_loader.py` — don't construct paths by hand.

FDR computation (`compute_fdr`) follows the SETS paper's **"1noisy"** convention: all noise-cluster (-1) misclassifications collapse to a single "fault", and `FDR = unique_faults_found / min(budget, total_faults)`. Don't change this — it's what the published numbers are measured against.

### Image pre-processing (load-bearing detail)

The SETS pretrained models (`SETS/Input_data/Pretrained_model/model_*.h5`) expect images scaled to **[-0.5, 0.5]**, *not* [0, 1]. Apply `X.astype("float32") / 255.0 - 0.5`. The `CLIP_MIN = -0.5, CLIP_MAX = 0.5` constants in `SETS/Source_code/cluster.py` are the hint.

Verification: load `model_mnist_LeNet1.h5`, run it on the MNIST test set with this pre-processing, and the predictions match the recorded `SETS/Input_data/Fault_clusters/mnist_LeNet1/mis_index_test.npy` (1542 misclassifications → acc 0.8458). With [0, 1] scaling instead, the model degrades to acc ≈ 0.5053 — a real bug that's easy to introduce. **Any new training / inference code in the robustness pipeline must use the [-0.5, 0.5] convention.**

Reference accuracies of the SETS pretrained models (test set, [-0.5, 0.5] pre-processing):

| Subject | Test accuracy | n_mis (out of 10 000) |
|---|---|---|
| `mnist_LeNet1` | 0.8458 | 1542 |
| `mnist_LeNet5` | 0.8785 | 1215 |

The retrained robustness instances should land within ±2 pp of these targets.

### QDPS algorithm (the load-bearing idea)

Quality-weighted DPP MAP inference with a **mixed similarity kernel**:

1. **Uncertainty (MaxP)**: `q_i = (1 - max(p_i))^quality_power`
2. **Reserve**: top `reserve_ratio*k` highest-uncertainty inputs are guaranteed selection (covers small-budget failure mode of pure DPP)
3. **Candidate pool**: top `alpha*k` by uncertainty (minus reserved)
4. **Kernel**: `K = feature_mix * K_vgg + (1 - feature_mix) * K_prob`, where `K_prob` is cosine similarity over **log-probability vectors** (this is the key novelty — behavioral diversity targets fault diversity directly).
5. **L-ensemble**: `L = diag(q) K diag(q)`, with a similarity penalty against the reserved set.
6. **Greedy MAP**: incremental Cholesky updates, O(Nk²).

Hyperparameters are **budget-adaptive** (see `ADAPTIVE_DEFAULTS` in `qdps.py`). Don't hardcode a single setting — the small/medium/large regimes were tuned per budget.

## Running on Compute Canada (Narval)

**From now on, every heavy step of the robustness pipeline — model training, VGG16 feature extraction, predict, and clustering — runs on Narval (Digital Research Alliance of Canada), not locally.** Local execution is reserved for development / smoke tests / inspecting small results.

### How Claude interacts with Narval

The user's terminal holds an interactive SSH session to Narval (`ssh narval`, which resolves via `~/.ssh/config` to `artem11@narval.alliancecan.ca`). That session also creates an SSH **ControlMaster socket** at `~/.ssh/cm-artem11@narval.alliancecan.ca:22` that persists for 1 hour after the last connection. Because of this socket, Claude can run remote commands directly from its `Bash` tool:

```
ssh narval '<remote command>'
```

This reuses the user's authenticated session — no passphrase prompts, no MFA prompts, no extra logins — as long as the user keeps a recent interactive SSH session alive (re-auth every ~1 hour). If the socket has expired, the user must reconnect interactively in their terminal first (`ssh narval`, handle MFA), which refreshes it.

**What Claude can do via this channel**: read/write small files, submit SLURM jobs (`sbatch`), check status (`squeue`, `sacct`), inspect outputs, clone/pull git. **What it cannot do**: long compute on the login node (against Alliance policy), interactive TTY apps (`htop`, `vim`).

### Filesystem layout on Narval

| Path | What's there | Why |
|---|---|---|
| `/home/artem11/QDPS/` | Git clone of the repo (code only) | `$HOME` is backed up but tiny (~50 GB). Code only. |
| `/scratch/artem11/QDPS/.venv/` | Python 3.10 virtualenv with Narval-wheelhouse packages | Active work; regenerable. |
| `/scratch/artem11/QDPS/instances/<subject>/seed_<n>/` | Trained `model.h5` + `meta.json` + `history.json` per (subject, seed) | Fast Lustre I/O for batch jobs. |
| `/scratch/artem11/QDPS/features_for_clustering/<dataset>/` | Raw VGG16 features (`features_{train,test}_raw.npy` per dataset subfolder) | 1 GB+ per dataset — must be on `$SCRATCH`. |
| `/scratch/artem11/QDPS/...` | All other per-instance outputs (predictions, cluster results) | Same reason. |
| `/project/def-manel131/artem11/` | (Not yet used) Final paper-ready artifacts | Backed up, persistent. Copy from `$SCRATCH` when ready to publish. |

**Symlink trick:** Because `train.py` and `extract_vgg_features.py` write to paths relative to themselves (`qdps/robustness/<subject>/instances/`), we create symlinks:

```
$HOME/QDPS/qdps/robustness/<subject>/instances  ->  $SCRATCH/QDPS/instances/<subject>
$HOME/QDPS/qdps/robustness/features_for_clustering  ->  $SCRATCH/QDPS/features_for_clustering
```

This makes the scripts write directly to `$SCRATCH` without any code change. Set these up after a fresh `git clone`.

### Python environment on Narval

Narval's wheelhouse pins differently from PyPI and uses `+computecanada` version suffixes. The **mac requirements (`qdps/robustness/requirements.txt`) do not all resolve on Narval** — for example `tensorflow==2.13.1` is not available; the earliest there is `2.15.1`.

The Narval install command Claude has tested working:

```bash
module load python/3.10
virtualenv --no-download $SCRATCH/QDPS/.venv
source $SCRATCH/QDPS/.venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index \
  "tensorflow~=2.15.0" \
  "scikit-learn~=1.3" \
  "h5py~=3.10" \
  "umap-learn>=0.5.7" \
  "hdbscan>=0.8.37" \
  matplotlib pandas scipy
```

Resolved versions: `tensorflow=2.15.1`, `keras=2.15.0`, `numpy=1.26.4`, `scipy=1.15.1`, `h5py=3.12.0`, `scikit-learn=1.5.2`, `hdbscan=0.8.37`, `matplotlib=3.10.0`, `pandas=2.2.3`.

Determinism preserved: seed 0 of `mnist_LeNet1` produces identical `val_accuracy` to the local Mac run despite different TF versions, because SGD with fixed seeds is deterministic across these environments.

### Running compute jobs

SLURM account: **`def-manel131`** (covers both CPU and GPU allocations).

Typical patterns Claude should use:

```bash
# Quick interactive check (login node is OK for ≤ ~2 min of CPU work)
ssh narval 'cd $HOME/QDPS && module load python/3.10 && source $SCRATCH/QDPS/.venv/bin/activate && python <script>.py'

# Real work — submit a SLURM batch job
ssh narval 'sbatch --account=def-manel131 <job_script.sh>'

# Check status
ssh narval 'squeue -u $USER'
ssh narval 'sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS'

# Job array for the 5 seeds in parallel
ssh narval 'sbatch --account=def-manel131 --array=0-4 <train_array.sh>'
```

**Approximate resource asks** (rules of thumb for our subjects):

| Step | GPU? | CPUs | RAM | Walltime |
|---|---|---|---|---|
| MNIST LeNet1/5 training (per seed) | No (CPU is fine) | 2 | 4 GB | 5 min |
| VGG16 feature extraction (per dataset) | 1× any GPU | 4 | 16 GB | 30 min |
| Per-instance predict.py | No | 2 | 4 GB | 5 min |
| Per-instance cluster.py (UMAP + HDBSCAN) | No | 8 | 16 GB | 30 min |
| Training larger subjects (CIFAR-10 / ResNet20) | 1× any GPU | 4 | 16 GB | 1 hr |
| Fruit-360 ResNet50 | 1× A100 | 8 | 32 GB | 6 hr |
| TinyImageNet ResNet101 | 1× A100 | 8 | 64 GB | 12 hr |

Login nodes only have CPUs — GPU work always goes through `sbatch`/`salloc`.

### When something on Narval breaks

1. **`Permission denied (keyboard-interactive,hostbased)`** when Claude calls `ssh narval` → the user's ControlMaster socket expired. Have them re-run `ssh narval` interactively, then retry.
2. **`No matching distribution found for X`** during `pip install --no-index` → wheelhouse version mismatch. Check what's available with `find /cvmfs/soft.computecanada.ca/custom/python/wheelhouse -name '<pkg>*'` and loosen the pin.
3. **Job stays in queue (`PD` state) for hours** → check `squeue -u $USER` and `sshare -U`. Account may be over its allocation; switch to the lower-priority "Resource Allocation Service" by dropping `--account` and Narval will use the default partition.

## Conventions

- Published baseline FDRs (SETS / DeepGD / RS paper Table 3 numbers) are duplicated in `qdps/BASELINES.md`, `qdps/run_single_subject.py`, `qdps/compare_results.py`, and `scripts/compare_results.py`. If any of them changes, update all four.
- Win/loss thresholds use a **±0.005 FDR band** for "tie" (see `compare_with_baseline` / `run_qdps_comparison`).
- Experiment outputs always include `meta` (method, timestamp, n_runs, budgets, subjects) plus per-run FDR/time arrays so results stay reproducible.
- The repo at `/Users/artem/Desktop/TCP` **is** a git repository now (initialized 2026-05-13, pushed to https://github.com/fekihatem11/QDPS). The vendored `SETS/` directory still has its own `.git/` — don't touch it.

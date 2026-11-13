# Retraining experiment, DeepGD protocol

Retraining experiment following the DeepGD paper (Aghababaeyan et al., TOSEM 2024,
RQ2), applied to the 8 SETS subjects (6 datasets). The earlier SETS-protocol
experiment (SETS RQ4) stays in the parent folder `qdps/retrain/`. See
`INVENTORY.md` for what the DeepGD, SETS and ATS packages provide.

## Generated test sets: decisions (2026-10-27)

| Decision | Choice |
|---|---|
| Generator | ATS `gen_data` (Gao et al., ICSE 2022), the code DeepGD used. Ported in `augment_ops.py` |
| Datasets | All 6: MNIST, Fashion-MNIST, CIFAR-10, SVHN, Fruit-360, TinyImageNet |
| Transformations | The 5 named in the DeepGD paper: rotation, zoom, shear, brightness, blur (no shift, no contrast) |
| Ranges | ATS's own per dataset. Fruit-360 and TinyImageNet have none in ATS, so they use CIFAR-10's (`augment_params.py`) |
| SVHN | ATS `SvhnDau` ranges, not DeepGD's separate OpenCV script, so all 6 datasets come from one generator |
| Seeds | One RNG per image, `default_rng([2026, op index, image index])`. Regeneration is bit-identical |
| Validity check | Sample grid + original-model accuracy per transformation, reviewed before retraining. No automatic threshold |

Each generated set holds one transformed copy of every test image per
transformation, so it is 5 × the test set. Image *i* of each file comes from test
image *i*, with the same label.

## Retraining: decisions (2026-10-27)

| Decision | Choice |
|---|---|
| Subjects | The 8 SETS subjects |
| Selection pool | The whole test set (valid indices), as in DeepGD. No T/V split |
| Budget | k = 300 (DeepGD's) |
| Methods | QDPS (final: product kernel, α = 3) and SETS |
| Selections | Precomputed once (`precompute_selections.py`, `.venv-timing`) into `selections_k300.json`; all 16 reproduce the official k = 300 FDRs. Clusters need no VGG features |
| Runs | 5 per condition (training seeds 0–4) |
| Training settings | Per subject, from `qdps/retrain/retrain_step.py` (identical to DeepGD's for the 6 small subjects) |
| Evaluation | Original test set T and combined generated set G (all 5 transformations) |
| Max-improvement runs | As the DeepGD paper (which produced its Table 7): FULL_T = training set + the whole test set; FULL_G = training set + the whole generated set. DeepGD's notebook (`Retraining.ipynb` cells 34, 39–40) fine-tunes on the inputs only; that definition gave effectiveness up to ~200% and was dropped on 2026-10-28, its runs are kept as `*_inputs_only`. Known issue: only MNIST LeNet5's retraining settings are published by DeepGD (cell 32); with them SVHN (both) and Fruit-360 (G) give negative maxima, so per-subject settings are being tuned |
| TinyImageNet | Best epoch chosen on T (as in the SETS/DeepGD tiny script); G reported at that epoch |

Outputs: `qdps/results/retrain_deepgd/subjects/<subject>.json` (per condition, per run),
`qdps/results/retrain_deepgd/TABLES.md` (improvements, optimization effectiveness,
Wilcoxon across subjects).

## Files

| File | Role |
|---|---|
| `augment_params.py` | Per-dataset ranges, transformation list, dataset → subjects |
| `augment_ops.py` | The 5 operators (ATS port; affine warp bit-identical to keras_preprocessing 1.1.2, brightness to `tf.image.adjust_brightness`) |
| `raw_test.py` | Raw uint8 test images and labels per dataset (same order and labels as `qdps/retrain/raw_data.py`) |
| `generate_augmented.py` | Writes `GENERATED_DATA_DIR/<dataset>/<OP>_test_{x,y}.npy` + `meta.json` |
| `check_augmented.py` | Writes `qdps/results/retrain_deepgd/generated_check/` (grids, `REPORT.md`, `accuracy.json`) |
| `precompute_selections.py`, `selections_k300.json` | The k = 300 QDPS and SETS selections, with their FDR |
| `protocol.py` | Protocol constants (k, methods, runs, conditions), combined generated set, model preprocessing |
| `backends.py` | Retrain + evaluate on T and G: Keras (7 subjects), PyTorch (TinyImageNet) |
| `run_retrain_deepgd.py` | One subject, a list of conditions; merges into the subject's results file (file-locked) |
| `report.py` | `TABLES.md` from all subjects' results |
| `fetch_results.sh <host>` | Copies results and job logs from a cluster into `qdps/results/retrain_deepgd/{subjects,logs,generated_check}` and rebuilds `TABLES.md` |
| `slurm/prepare_large_datasets.sh` | Fruit-360 arrays + Fruit-360/TinyImageNet generated sets + check |
| `slurm/retrain_subject.sh`, `slurm/submit_all.sh` | One (subject, conditions) job; all 32 jobs behind a mnist_LeNet1 pilot |

`GENERATED_DATA_DIR` = `qdps/datasets/retrain/generated/` locally, or
`$QDPS_RETRAIN_DIR/generated/` on Narval.

## Commands (repo root, `.venv`)

```
python -m qdps.retrain.retrain_deepgd_version.generate_augmented mnist fashion_mnist cifar10 svhn
python -m qdps.retrain.retrain_deepgd_version.check_augmented mnist fashion_mnist cifar10 svhn
# Narval only (raw data lives there):
python -m qdps.retrain.retrain_deepgd_version.generate_augmented fruit360 tinyimagenet
# Retraining (Narval):
bash qdps/retrain/retrain_deepgd_version/slurm/submit_all.sh <prepare_job_id>
python -m qdps.retrain.retrain_deepgd_version.report
```

Extra dependency: OpenCV 4.10 (blur filters): `opencv-python-headless==4.10.0.84` in
`.venv`, `module load gcc/12.3 opencv/4.10.0` on Narval.

**Official copies = Narval's** (`/project/def-manel131/artem11/QDPS/retrain/generated/`),
since all retraining runs there. Mac and Narval agree bit for bit on rotation, zoom, shear
and brightness. Blur differs only through the bilateral filter (different OpenCV builds):
±1 intensity level on 1 Fashion-MNIST and 37 CIFAR-10 images.

## Status

| Dataset | Generated | Checked |
|---|---|---|
| MNIST, Fashion-MNIST, CIFAR-10, SVHN | yes (Mac and Narval) | yes, see `generated_check/REPORT.md` |
| Fruit-360, TinyImageNet | queued: `slurm/prepare_large_datasets.sh`, job 7499560 (Narval power outage until 2026-11-03) | no |

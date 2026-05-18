# `qdps/robustness/slurm/`

SLURM batch scripts for running the robustness pipeline on Narval (DRAC).
Grouped by pipeline stage.

## Layout

```
slurm/
├── train/        # Per-subject training (5 instances per subject)
│   └── train_mnist_lenet1.sh           # array 0-4
├── features/     # One-time VGG16 feature extraction per dataset
│   └── extract_vgg_features.sh         # single GPU job
├── predict/      # Per-instance softmax + misclassification indices
│   └── predict.sh                      # array 0-4
└── cluster/      # Fault clustering
    ├── cluster_sweep.sh                # Sequential 80-config sweep (fallback)
    ├── cluster_sweep_array.sh          # Parallel sweep: 80-task array
    ├── aggregate_sweep.sh              # Post-array aggregator
    └── cluster.sh                      # Per-instance clustering w/ LOCKED_CONFIG
                                         # (array 0-4 after sweep -> winner picked)
```

## Pipeline order

```
1.  train/train_mnist_lenet1.sh                 -> 5 instances
2.  features/extract_vgg_features.sh            -> per-dataset features
3.  predict/predict.sh                          -> per-instance softmax + mis-indices
4.  cluster/cluster_sweep_array.sh              -> 80 per-config JSONs (seed 0 only)
    + cluster/aggregate_sweep.sh                -> sweep_results.json -> WINNER
5.  hand-update LOCKED_CONFIG in cluster.py with the winner, commit, push
6.  cluster/cluster.sh                          -> 5 cluster_results.npy
```

## Common conventions

- All scripts hard-code `--account=def-manel131`.
- Subject selection via env var: `--export=ALL,SUBJECT=mnist_LeNet1`.
- Dataset selection via env var: `--export=ALL,DATASET=mnist`.
- Email notifications: `END` + `FAIL` (arrays add `ARRAY_TASKS`).
- Logs land in `$SCRATCH/QDPS/slurm_logs/<jobname>-<jobid>[_<task>].{out,err}`.
- Outputs land in `$SCRATCH/QDPS/instances/...` via the per-subject and
  per-dataset symlinks set up at clone time (see `CLAUDE.md`).

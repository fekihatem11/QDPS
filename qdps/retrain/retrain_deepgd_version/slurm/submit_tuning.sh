#!/bin/bash
# Step 2: tuning runs for the retraining settings (Fir), 2026-10-28. Candidates and rule
# in tune_report.py. Logs: $SCRATCH/QDPS/logs/tune-<subject>-<tag>-<jobid>.out
#   cd $HOME/QDPS && bash qdps/retrain/retrain_deepgd_version/slurm/submit_tuning.sh
set -euo pipefail
cd $HOME/QDPS
source qdps/retrain/slurm/env.sh  # cluster settings from .env (exports SBATCH_ACCOUNT)
mkdir -p $SCRATCH/QDPS/logs
S=qdps/retrain/retrain_deepgd_version/slurm/tune_settings.sh
MIG=nvidia_h100_80gb_hbm3_3g.40gb:1

sub() {  # <gpu> <mem> <time> <name> <tune args...>
    local gpu=$1 mem=$2 time=$3 name=$4; shift 4
    sbatch --gpus-per-node=$gpu --mem=$mem --time=$time --job-name=tune-$name \
        --output=$SCRATCH/QDPS/logs/%x-%j.out $S "$@"
}

for subject in Fashion_mnist_LeNet4 cifar10_12Conv cifar10_ResNet20 SVHN_LeNet5; do
    for opt_lr in adadelta:0.005 adadelta:0.05 adam:0.0001 adadelta:1.0 adam:0.001; do
        opt=${opt_lr%%:*}; lr=${opt_lr##*:}
        sub $MIG 64G 3:00:00 $subject-$opt-$lr $subject $opt $lr
    done
done
for lr in 0.00001 0.0001; do
    sub h100:1 160G 6:00:00 Fruit360_ResNet50-adam-$lr-logits Fruit360_ResNet50 adam $lr --from-logits
done

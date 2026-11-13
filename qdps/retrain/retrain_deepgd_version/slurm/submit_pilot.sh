#!/bin/bash
# Pilot of retraining settings for the negative max-improvement runs (Fir), 2026-10-28.
#   cd $HOME/QDPS && bash qdps/retrain/retrain_deepgd_version/slurm/submit_pilot.sh
# Logs go to $SCRATCH/QDPS/logs/pilot-<subject>-<condition>-<tag>-<jobid>.out
set -euo pipefail
cd $HOME/QDPS
source qdps/retrain/slurm/env.sh  # cluster settings from .env (exports SBATCH_ACCOUNT)
mkdir -p $SCRATCH/QDPS/logs
S=qdps/retrain/retrain_deepgd_version/slurm/pilot_settings.sh
MIG=nvidia_h100_80gb_hbm3_3g.40gb:1

sub() {  # <gpu> <mem> <time> <name> <pilot args...>
    local gpu=$1 mem=$2 time=$3 name=$4; shift 4
    sbatch --gpus-per-node=$gpu --mem=$mem --time=$time --job-name=pilot-$name \
        --output=$SCRATCH/QDPS/logs/%x-%j.out $S "$@"
}

# SVHN LeNet5: current setting (Adadelta 0.005) as reference, then stronger optimizers.
for cond in FULL_T FULL_G; do
    mem=32G; [ $cond = FULL_G ] && mem=64G
    for opt_lr in adadelta:0.005 adadelta:0.05 adadelta:1.0 adam:0.0001 adam:0.001; do
        opt=${opt_lr%%:*}; lr=${opt_lr##*:}
        sub $MIG $mem 2:00:00 SVHN-$cond-$opt-$lr SVHN_LeNet5 $cond $opt $lr
    done
done

# Fruit-360 ResNet50: current setting (Adam 1e-5, from_logits=False) as reference, then
# the corrected loss (from_logits=True) at two learning rates.
sub h100:1 160G 4:00:00 Fruit-FULL_G-adam-1e-5        Fruit360_ResNet50 FULL_G adam 0.00001
sub h100:1 160G 4:00:00 Fruit-FULL_G-adam-1e-5-logits Fruit360_ResNet50 FULL_G adam 0.00001 --from-logits
sub h100:1 160G 4:00:00 Fruit-FULL_G-adam-1e-4-logits Fruit360_ResNet50 FULL_G adam 0.0001 --from-logits
sub h100:1 128G 3:00:00 Fruit-FULL_T-adam-1e-5-logits Fruit360_ResNet50 FULL_T adam 0.00001 --from-logits

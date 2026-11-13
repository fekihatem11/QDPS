#!/bin/bash
# Submit the RQ4 (SETS-protocol) retraining for all 8 subjects, k = 500, 5 runs, Fir.
# START_RUN=5 bash ... submits runs 6-10 (seeds 5-9), stored apart (subjects_runs5-9/).
# The 6 small subjects run QDPS and SETS in one job; Fruit-360 and TinyImageNet run one
# method per job. Logs: $SCRATCH/QDPS/logs/rq4-<subject>-<methods>-<jobid>.out
#   cd $HOME/QDPS && bash qdps/retrain/slurm/submit_rq4.sh
set -euo pipefail
cd $HOME/QDPS
LOGS=$SCRATCH/QDPS/logs
mkdir -p $LOGS
S=qdps/retrain/slurm/retrain_subject.sh
MIG=nvidia_h100_80gb_hbm3_3g.40gb:1

START_RUN=${START_RUN:-0}
sub() {  # <subject> <methods> <gpu> <mem> <time>
    sbatch --account=def-manel131 --job-name=rq4-$1-${2/,/+}-r$START_RUN --gpus-per-node=$3 --mem=$4 --time=$5 \
        --output=$LOGS/%x-%j.out $S "$1" 500 5 "$2" $START_RUN
}
for s in mnist_LeNet1 mnist_LeNet5 Fashion_mnist_LeNet4; do sub $s QDPS,SETS $MIG 32G 4:00:00; done
for s in cifar10_12Conv cifar10_ResNet20 SVHN_LeNet5; do sub $s QDPS,SETS $MIG 48G 8:00:00; done
for m in QDPS SETS; do
    sub Fruit360_ResNet50 $m h100:1 128G 12:00:00
    sub TinyImageNet_ResNet101 $m h100:1 64G 16:00:00
done

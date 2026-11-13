#!/bin/bash
# Submit the whole DeepGD-protocol retraining: one job per (subject, condition), 5 runs each.
#
#   bash qdps/retrain/retrain_deepgd_version/slurm/submit_all.sh [prep_job_id]
#
# GPU request: GPU_SMALL (LeNets, CIFAR-10 models, SVHN) and GPU_LARGE (Fruit-360,
# TinyImageNet), default "1" (Narval). Fir needs a type, e.g.
#   GPU_SMALL=nvidia_h100_80gb_hbm3_3g.40gb:1 GPU_LARGE=h100:1 bash .../submit_all.sh <prep_job_id>
#
# mnist_LeNet1 (all 4 conditions) runs first as a pilot; every other job waits for it
# to succeed (afterok), so a bug cannot burn GPU hours across 32 jobs. Fruit-360 and
# TinyImageNet also wait for the preparation job (prepare_large_datasets.sh) if its id
# is given. Logs go to $SCRATCH/QDPS/logs.
# SKIP_PILOT=1: the pilot already succeeded; submit only the other jobs, without it.
# SUBJECTS="..." / CONDITIONS="...": override the subject list (default: all but the pilot
# subject) and the conditions (default: QDPS SETS FULL_T FULL_G), e.g. to rerun FULL_T FULL_G.
set -euo pipefail
cd $HOME/QDPS
PREP_JOB="${1:-}"
GPU_SMALL="${GPU_SMALL:-1}"
GPU_LARGE="${GPU_LARGE:-1}"
SCRIPT=qdps/retrain/retrain_deepgd_version/slurm/retrain_subject.sh
LOGS=$SCRATCH/QDPS/logs
mkdir -p $LOGS

# subject -> "mem time" for QDPS/SETS/FULL_T, and for FULL_G (larger training set)
declare -A RES=(
    [mnist_LeNet1]="24G 06:00:00|32G 10:00:00"
    [mnist_LeNet5]="24G 06:00:00|32G 10:00:00"
    [Fashion_mnist_LeNet4]="24G 06:00:00|32G 10:00:00"
    [cifar10_12Conv]="32G 12:00:00|48G 24:00:00"
    [cifar10_ResNet20]="32G 12:00:00|48G 24:00:00"
    [SVHN_LeNet5]="32G 08:00:00|64G 16:00:00"
    [Fruit360_ResNet50]="128G 12:00:00|160G 36:00:00"
    [TinyImageNet_ResNet101]="64G 24:00:00|64G 48:00:00"
)

submit() {  # subject condition dependency
    local spec=${RES[$1]}
    [ "$2" = "FULL_G" ] && spec=${spec#*|} || spec=${spec%|*}
    local mem=${spec% *} time=${spec#* } gpu=$GPU_SMALL
    [[ $1 == Fruit360_ResNet50 || $1 == TinyImageNet_ResNet101 ]] && gpu=$GPU_LARGE
    sbatch --parsable --chdir=$LOGS --job-name="rt-$1-$2" --mem=$mem --time=$time --gpus-per-node=$gpu \
        ${3:+--dependency=$3} $SCRIPT "$1" "$2" 5
}

if [ "${SKIP_PILOT:-0}" = "1" ]; then
    PILOT=""
    echo "pilot skipped (already done)"
else
    PILOT=$(sbatch --parsable --chdir=$LOGS --job-name="rt-mnist_LeNet1-pilot" --mem=32G --time=16:00:00 --gpus-per-node=$GPU_SMALL \
        $SCRIPT mnist_LeNet1 QDPS,SETS,FULL_T,FULL_G 5)
    echo "pilot mnist_LeNet1: $PILOT"
fi
SUBJECTS="${SUBJECTS:-mnist_LeNet5 Fashion_mnist_LeNet4 cifar10_12Conv cifar10_ResNet20 SVHN_LeNet5 Fruit360_ResNet50 TinyImageNet_ResNet101}"
CONDITIONS="${CONDITIONS:-QDPS SETS FULL_T FULL_G}"
for subject in $SUBJECTS; do
    deps=()
    [ -n "$PILOT" ] && deps+=("$PILOT")
    if [ -n "$PREP_JOB" ] && [[ $subject == Fruit360_ResNet50 || $subject == TinyImageNet_ResNet101 ]]; then
        deps+=("$PREP_JOB")
    fi
    dep=""
    [ ${#deps[@]} -gt 0 ] && dep="afterok:$(IFS=:; echo "${deps[*]}")"
    for condition in $CONDITIONS; do
        echo "$subject $condition: $(submit $subject $condition $dep)"
    done
done
squeue -u $USER

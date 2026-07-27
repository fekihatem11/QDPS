#!/bin/bash
# RQ4 retraining experiment on Narval — one subject per job.
# Usage: sbatch --account=def-manel131 retrain_subject.sh <subject_key> [k] [n_runs]
#   e.g. sbatch --account=def-manel131 retrain_subject.sh cifar10_12Conv 500 5
#
# Data layout expected on $SCRATCH (synced from the dev machine, all gitignored):
#   $SCRATCH/QDPS/retrain/{pretrained,splits}/         (QDPS_RETRAIN_DIR)
#   $SCRATCH/QDPS/keras_cache/datasets/                (KERAS_HOME - offline keras data)
#   ~/QDPS/qdps/datasets/{fault_clusters,features_for_selection}/  (symlinks -> $SCRATCH)
#SBATCH --job-name=qdps-retrain
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

SUBJECT="${1:?usage: retrain_subject.sh <subject_key> [k] [n_runs]}"
K="${2:-500}"
N_RUNS="${3:-5}"

module load StdEnv/2023 python/3.10 cuda cudnn

source "$SCRATCH/QDPS/.venv/bin/activate"
export KERAS_HOME="$SCRATCH/QDPS/keras_cache"
export QDPS_RETRAIN_DIR="$SCRATCH/QDPS/retrain"

cd "$HOME/QDPS"
echo "subject=$SUBJECT k=$K n_runs=$N_RUNS  node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-none}"
python -u -m qdps.retrain.run_retrain "$SUBJECT" "$K" "$N_RUNS"

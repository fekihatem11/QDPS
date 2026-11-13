#!/bin/bash
# RQ4 retraining experiment (SETS protocol) — one subject per job, on Fir or Narval.
#
#   sbatch --account=def-manel131 --gpus-per-node=<gpu> [--mem=.. --time=..] \
#       qdps/retrain/slurm/retrain_subject.sh <subject_key> [k] [n_runs] [methods] [start_run]
#
# start_run > 0 (e.g. 5 = runs 6-10) stores into results/retrain/subjects_runs<a>-<b>/.
#
# Selections come from qdps/retrain/selections_k<k>.json (precompute_selections.py, run
# on the dev machine), so the nodes need no VGG features. Results merge into
# $HOME/QDPS/qdps/results/retrain/subjects/<subject>.json (one-method jobs accumulate).
# Layout (2026-10-27): venv and retrain assets on /project; TinyImageNet images are
# unpacked from the archive to node-local disk.
#SBATCH --job-name=qdps-retrain
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

SUBJECT="${1:?usage: retrain_subject.sh <subject_key> [k] [n_runs] [methods]}"
K="${2:-500}"
N_RUNS="${3:-5}"
METHODS="${4:-QDPS,SETS}"
START_RUN="${5:-0}"
PROJECT=/project/def-manel131/artem11/QDPS
export QDPS_RETRAIN_DIR=$PROJECT/retrain

module load StdEnv/2023 python/3.10 gcc/12.3 opencv/4.10.0 cuda cudnn
source $PROJECT/.venv/bin/activate
cd $HOME/QDPS
echo "subject=$SUBJECT k=$K n_runs=$N_RUNS methods=$METHODS start_run=$START_RUN node=$(hostname) commit=$(git rev-parse --short HEAD)"

if [ "$SUBJECT" = "TinyImageNet_ResNet101" ]; then
    mkdir -p $SLURM_TMPDIR/raw
    unzip -q $QDPS_RETRAIN_DIR/raw/archives/tiny-imagenet-200.zip -d $SLURM_TMPDIR/raw
    export QDPS_RAW_DATA_DIR=$SLURM_TMPDIR/raw
fi

python -u -m qdps.retrain.run_retrain "$SUBJECT" "$K" "$N_RUNS" "$METHODS" "$START_RUN"

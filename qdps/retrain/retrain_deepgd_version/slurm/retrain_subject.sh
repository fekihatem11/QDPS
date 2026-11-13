#!/bin/bash
#SBATCH --job-name=retrain-deepgd
#SBATCH --account=def-manel131
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# DeepGD-protocol retraining for one subject and a list of conditions (Narval).
#
#   sbatch [--mem=.. --time=..] qdps/retrain/retrain_deepgd_version/slurm/retrain_subject.sh <subject> [conditions] [n_runs]
#
# Conditions: QDPS,SETS,FULL_T,FULL_G (default all). Results merge into
# $HOME/QDPS/qdps/results/retrain_deepgd/subjects/<subject>.json, so a subject's
# conditions can run as parallel jobs. Fruit-360 FULL_G needs ~128G (train + 118k
# generated images in float32); TinyImageNet needs its archive unpacked (done below).
set -euo pipefail

SUBJECT="${1:?usage: retrain_subject.sh <subject> [conditions] [n_runs]}"
CONDITIONS="${2:-QDPS,SETS,FULL_T,FULL_G}"
N_RUNS="${3:-5}"
PROJECT=/project/def-manel131/artem11/QDPS
export QDPS_RETRAIN_DIR=$PROJECT/retrain

module load StdEnv/2023 python/3.10 gcc/12.3 opencv/4.10.0 cuda cudnn
source $PROJECT/.venv/bin/activate
cd $HOME/QDPS
echo "subject=$SUBJECT conditions=$CONDITIONS n_runs=$N_RUNS node=$(hostname) commit=$(git rev-parse --short HEAD)"

if [ "$SUBJECT" = "TinyImageNet_ResNet101" ]; then
    mkdir -p $SLURM_TMPDIR/raw
    unzip -q $QDPS_RETRAIN_DIR/raw/archives/tiny-imagenet-200.zip -d $SLURM_TMPDIR/raw
    export QDPS_RAW_DATA_DIR=$SLURM_TMPDIR/raw
fi

python -u -m qdps.retrain.retrain_deepgd_version.run_retrain_deepgd "$SUBJECT" "$CONDITIONS" "$N_RUNS"

#!/bin/bash
#SBATCH --job-name=prep-fruit-tiny
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Prepare Fruit-360 and TinyImageNet for the DeepGD-protocol retraining (Narval).
#
# 1. Fruit-360: unpack the fruits-360-100x100 snapshot (commit 2f981c8) to node-local
#    disk, rebuild the npy arrays into $QDPS_RETRAIN_DIR/raw/fruit360 and verify them
#    against the recorded misclassified set (build_fruit_npy.py prints EXACT MATCH).
# 2. Generate the transformed test sets of both datasets into $QDPS_RETRAIN_DIR/generated.
# 3. Validity check for Fruit-360 (TinyImageNet is a torch model: checked separately).
#
# Submit from the repo root:  sbatch qdps/retrain/retrain_deepgd_version/slurm/prepare_large_datasets.sh
set -euo pipefail

source "${QDPS_REPO_DIR:-$HOME/QDPS}/qdps/retrain/slurm/env.sh"  # cluster settings from .env
PROJECT=$QDPS_PROJECT_DIR
ARCHIVES=$PROJECT/retrain/raw/archives
export QDPS_RETRAIN_DIR=$PROJECT/retrain

module load StdEnv/2023 python/3.10 gcc/12.3 opencv/4.10.0 cuda cudnn
source $PROJECT/.venv/bin/activate
cd $HOME/QDPS
echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} commit=$(git rev-parse --short HEAD)"

# ---- 1. Fruit-360 arrays (skipped if already built) ----
if [ ! -f $QDPS_RETRAIN_DIR/raw/fruit360/fruit_y_test.npy ]; then
    tar -xzf $ARCHIVES/fruits-360-100x100-2f981c8.tar.gz -C $SLURM_TMPDIR
    FRUIT_ROOT=$(ls -d $SLURM_TMPDIR/fruits-360-100x100-*)
    python -u -m qdps.retrain.build_fruit_npy $FRUIT_ROOT
fi

# ---- 2. Generated test sets ----
# TinyImageNet images are read from node-local disk; Fruit-360 from its npy arrays.
unzip -q $ARCHIVES/tiny-imagenet-200.zip -d $SLURM_TMPDIR/raw
ln -s $QDPS_RETRAIN_DIR/raw/fruit360 $SLURM_TMPDIR/raw/fruit360
export QDPS_RAW_DATA_DIR=$SLURM_TMPDIR/raw
python -u -m qdps.retrain.retrain_deepgd_version.generate_augmented fruit360 tinyimagenet

# ---- 3. Validity check (Fruit-360) ----
python -u -m qdps.retrain.retrain_deepgd_version.check_augmented fruit360

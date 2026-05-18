#!/bin/bash
# Extract VGG16 (block5_conv3) features for a dataset's train + test splits.
# One-time per dataset; output reused by all retrained instances.
# Writes $SCRATCH/QDPS/features_for_clustering/<dataset>/features_{train,test}_raw.npy
# via the symlink $HOME/QDPS/qdps/robustness/features_for_clustering -> $SCRATCH/...
#
# Submit:   sbatch --export=ALL,DATASET=mnist qdps/robustness/slurm/extract_vgg_features.sh
# Default:  DATASET=mnist
# Status:   squeue -u $USER
# Logs:     $SCRATCH/QDPS/slurm_logs/vgg-extract-<jobid>.out

#SBATCH --account=def-manel131
#SBATCH --job-name=vgg-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/QDPS/slurm_logs/vgg-extract-%j.out
#SBATCH --error=/scratch/%u/QDPS/slurm_logs/vgg-extract-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fekihatem72@gmail.com

set -e

module purge
module load python/3.10 cuda cudnn

source "$SCRATCH/QDPS/.venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_CPP_MIN_LOG_LEVEL=2

cd "$HOME/QDPS"

DATASET=${DATASET:-mnist}

echo "----- SLURM job info -----"
echo "Job:      $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Host:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Started:  $(date -Iseconds)"
echo "Commit:   $(git rev-parse --short HEAD)"
echo "Dataset:  $DATASET"
echo "Python:   $(python --version)"
echo "--------------------------"

python qdps/robustness/extract_vgg_features.py --dataset "$DATASET"

echo "Finished: $(date -Iseconds)"

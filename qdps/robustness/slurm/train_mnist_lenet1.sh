#!/bin/bash
# Train 5 LeNet1/MNIST robustness instances on Narval as a SLURM job array.
# Each task trains one seed (0..4) independently and writes
# $SCRATCH/QDPS/instances/mnist_LeNet1/seed_<n>/{model.h5,meta.json,history.json}
# via the symlink $HOME/QDPS/qdps/robustness/mnist_LeNet1/instances -> $SCRATCH/...
#
# Submit:   sbatch qdps/robustness/slurm/train_mnist_lenet1.sh
# Status:   squeue -u $USER
# Logs:     $SCRATCH/QDPS/slurm_logs/lenet1-<jobid>_<task>.out

#SBATCH --account=def-manel131
#SBATCH --job-name=lenet1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/QDPS/slurm_logs/lenet1-%A_%a.out
#SBATCH --error=/scratch/%u/QDPS/slurm_logs/lenet1-%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=fekihatem72@gmail.com

set -e

module purge
module load python/3.10

source "$SCRATCH/QDPS/.venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_CPP_MIN_LOG_LEVEL=2

cd "$HOME/QDPS"

echo "----- SLURM job info -----"
echo "Job:      $SLURM_JOB_NAME ($SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID)"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "Commit:   $(git rev-parse --short HEAD)"
echo "Python:   $(python --version)"
echo "TF threads (OMP_NUM_THREADS): $OMP_NUM_THREADS"
echo "--------------------------"

python qdps/robustness/mnist_LeNet1/train.py --seed "$SLURM_ARRAY_TASK_ID"

echo "Finished: $(date -Iseconds)"

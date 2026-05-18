#!/bin/bash
# Per-instance prediction for a robustness subject.
# Runs as a 5-task job array (one per seed) for the given SUBJECT.
# Writes $SCRATCH/QDPS/instances/<subject>/seed_<n>/predictions/*
# via the symlink $HOME/QDPS/qdps/robustness/<subject>/instances -> $SCRATCH/...
#
# Submit (default subject=mnist_LeNet1):
#     sbatch qdps/robustness/slurm/predict.sh
# Subject override:
#     sbatch --export=ALL,SUBJECT=mnist_LeNet5 qdps/robustness/slurm/predict.sh
# Logs: $SCRATCH/QDPS/slurm_logs/predict-<jobid>_<task>.out

#SBATCH --account=def-manel131
#SBATCH --job-name=predict
#SBATCH --array=0-4
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/QDPS/slurm_logs/predict-%A_%a.out
#SBATCH --error=/scratch/%u/QDPS/slurm_logs/predict-%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=fekihatem72@gmail.com

set -e

module purge
module load python/3.10

source "$SCRATCH/QDPS/.venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_CPP_MIN_LOG_LEVEL=2

cd "$HOME/QDPS"

SUBJECT=${SUBJECT:-mnist_LeNet1}

echo "----- SLURM job info -----"
echo "Job:      $SLURM_JOB_NAME ($SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID)"
echo "Host:     $(hostname)"
echo "Subject:  $SUBJECT"
echo "Seed:     $SLURM_ARRAY_TASK_ID"
echo "Started:  $(date -Iseconds)"
echo "Commit:   $(git rev-parse --short HEAD)"
echo "Python:   $(python --version)"
echo "--------------------------"

python qdps/robustness/predict.py --subject "$SUBJECT" --seed "$SLURM_ARRAY_TASK_ID"

echo "Finished: $(date -Iseconds)"
